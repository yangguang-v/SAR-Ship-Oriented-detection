# gaussian_fcos_head_decoupled_gaucho.py
# Copyright (c) OpenMMLab. All rights reserved.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean

from mmcv.ops import box_iou_rotated

from mmrotate.core import multiclass_nms_rotated

from ..builder import ROTATED_HEADS, ROTATED_LOSSES, build_loss
from .rotated_fcos_head import RotatedFCOSHead
from ...core.bbox.transforms import norm_angle


INF = 1e8
EPS = 1e-6


# ==============================================================
# Basic Gaussian utilities
# ==============================================================

def cholesky_to_covariance(cholesky):
    """
    Cholesky factor:
        L = [[a, 0],
             [c, b]]

    Sigma = L L^T

    Args:
        cholesky: (..., 3), [a, b, c]

    Returns:
        cov_a, cov_b, cov_c
        Sigma =
            [[cov_a, cov_c],
             [cov_c, cov_b]]
    """
    a = cholesky[..., 0]
    b = cholesky[..., 1]
    c = cholesky[..., 2]

    cov_a = a.square()
    cov_b = b.square() + c.square()
    cov_c = a * c

    return cov_a, cov_b, cov_c


def decode_gaucho_offset(
        points,
        deltas,
        angle_range,
        to_obb=False):
    """
    Decode GauCho geometry branch.

    deltas:
        [dx, dy, a, b, c]

    Returns:
        to_obb=False:
            [cx, cy, cov_a, cov_b, cov_c]

        to_obb=True:
            [cx, cy, w, h, angle]
    """

    ctr = deltas[..., :2] + points

    cholesky = deltas[..., 2:5]

    cov_a, cov_b, cov_c = cholesky_to_covariance(
        cholesky
    )

    if to_obb:

        eig_gap = torch.sqrt(
            torch.clamp(
                4.0 * cov_c.square()
                + (cov_a - cov_b).square(),
                min=1e-9
            )
        )

        eig_small = (
            0.5 * (
                cov_a
                + cov_b
                - eig_gap
            )
        ).clamp(min=1e-8)

        eig_large = (
            0.5 * (
                cov_a
                + cov_b
                + eig_gap
            )
        ).clamp(min=1e-8)

        gw = (
            2.0 * torch.sqrt(eig_large)
        ).unsqueeze(-1)

        gh = (
            2.0 * torch.sqrt(eig_small)
        ).unsqueeze(-1)

        gt = (
            0.5
            * torch.atan2(
                2.0 * cov_c,
                cov_a - cov_b + 1e-6
            )
        ).unsqueeze(-1)

        gt = norm_angle(
            gt,
            angle_range
        )

        return torch.cat(
            [
                ctr,
                gw,
                gh,
                gt
            ],
            dim=-1
        )

    return torch.cat(
        [
            ctr,
            cov_a.unsqueeze(-1),
            cov_b.unsqueeze(-1),
            cov_c.unsqueeze(-1)
        ],
        dim=-1
    )


def covariance_to_matrix(
        cov_a,
        cov_b,
        cov_c):
    """
    Build symmetric 2-D covariance matrix.
    """

    sigma = torch.stack(
        [
            torch.stack(
                [cov_a, cov_c],
                dim=-1
            ),
            torch.stack(
                [cov_c, cov_b],
                dim=-1
            )
        ],
        dim=-2
    )

    return sigma


def xywhr_to_gaussian_sigma(xywhr):
    """Convert rotated boxes [cx, cy, w, h, theta] to 2-D Gaussian covariance.

    The Gaussian representation follows the same convention as GauCho:
        S = diag(w/2, h/2)
        Sigma = R S^2 R^T

    This makes the uncertainty-aware NLL supervise the complete Gaussian
    geometry corresponding to center, scale, and orientation of the OBB.
    """
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-4)
    theta = xywhr[..., 4]

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    r00 = cos_t
    r01 = -sin_t
    r10 = sin_t
    r11 = cos_t

    sx2 = 0.25 * wh[..., 0].square()
    sy2 = 0.25 * wh[..., 1].square()

    sigma_xx = r00.square() * sx2 + r01.square() * sy2
    sigma_yy = r10.square() * sx2 + r11.square() * sy2
    sigma_xy = r00 * r10 * sx2 + r01 * r11 * sy2

    sigma = torch.stack(
        [
            torch.stack([sigma_xx, sigma_xy], dim=-1),
            torch.stack([sigma_xy, sigma_yy], dim=-1)
        ],
        dim=-2
    )

    return xy, sigma


def gaussian_geometry_nll_2d(
        pred_mean,
        pred_sigma,
        target_mean,
        target_sigma,
        weight=None,
        avg_factor=None):
    """Expected Gaussian NLL for complete 2-D geometry.

    Target geometry is treated as a Gaussian distribution: 
        X_gt ~ N(mu_gt, Sigma_gt)

    The predicted distribution is the uncertainty-coupled Gaussian:
        X_pred ~ N(mu_pred, Sigma_g + lambda_c Sigma_c)

    Ignoring the constant 0.5*d*log(2*pi), the expected NLL is:

        0.5 * [
            (mu_gt-mu_pred)^T Sigma^-1 (mu_gt-mu_pred)
            + tr(Sigma^-1 Sigma_gt)
            + log |Sigma|
        ]

    Unlike a center-only NLL, this term directly couples:
        center + width/height + orientation.
    """
    eye = torch.eye(2, device=pred_sigma.device, dtype=pred_sigma.dtype)
    eye = eye.unsqueeze(0)

    sigma = pred_sigma + eye * EPS

    sxx = sigma[:, 0, 0]
    sxy = sigma[:, 0, 1]
    syy = sigma[:, 1, 1]

    det = (sxx * syy - sxy.square()).clamp(min=EPS)

    inv_xx = syy / det
    inv_yy = sxx / det
    inv_xy = -sxy / det

    delta = target_mean - pred_mean
    dx = delta[:, 0]
    dy = delta[:, 1]

    mahal = (
        inv_xx * dx.square()
        + inv_yy * dy.square()
        + 2.0 * inv_xy * dx * dy
    ).clamp(min=0.0)

    txx = target_sigma[:, 0, 0]
    txy = target_sigma[:, 0, 1]
    tyy = target_sigma[:, 1, 1]

    trace_term = (
        inv_xx * txx
        + 2.0 * inv_xy * txy
        + inv_yy * tyy
    ).clamp(min=0.0)

    nll = 0.5 * (
        mahal
        + trace_term
        + torch.log(det)
    )

    if weight is not None:
        nll = nll * weight

    if avg_factor is not None:
        return nll.sum() / avg_factor

    return nll.mean()


def gaussian_nll_2d(
        residual,
        sigma,
        weight=None,
        avg_factor=None):
    """
    Zero-mean 2-D Gaussian negative log-likelihood.

        e ~ N(0, Sigma)

        NLL = 0.5[
              e^T Sigma^-1 e
              + log |Sigma|
        ]

    Args:
        residual: (N, 2)
        sigma:    (N, 2, 2)

    Returns:
        scalar loss
    """

    eye = torch.eye(
        2,
        device=sigma.device,
        dtype=sigma.dtype
    ).unsqueeze(0)

    sigma = sigma + eye * EPS

    sxx = sigma[:, 0, 0]
    sxy = sigma[:, 0, 1]
    syy = sigma[:, 1, 1]

    det = (
        sxx * syy
        - sxy.square()
    ).clamp(min=EPS)

    inv_xx = syy / det
    inv_yy = sxx / det
    inv_xy = -sxy / det

    rx = residual[:, 0]
    ry = residual[:, 1]

    mahal = (
        inv_xx * rx.square()
        + inv_yy * ry.square()
        + 2.0 * inv_xy * rx * ry
    )

    mahal = mahal.clamp(min=0.0)

    nll = 0.5 * (
        mahal
        + torch.log(det)
    )

    if weight is not None:
        nll = nll * weight

    if avg_factor is not None:
        return nll.sum() / avg_factor

    return nll.mean()


# ==============================================================
# High-IoU Gaussian geometry loss
# ==============================================================

def _xywhr_to_gaussian_sigma_for_loss(xywhr):
    """Convert OBB [cx, cy, w, h, theta] to 2-D Gaussian covariance."""
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-4)
    theta = xywhr[..., 4]

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    sx2 = 0.25 * wh[..., 0].square()
    sy2 = 0.25 * wh[..., 1].square()

    sigma_xx = cos_t.square() * sx2 + sin_t.square() * sy2
    sigma_yy = sin_t.square() * sx2 + cos_t.square() * sy2
    sigma_xy = sin_t * cos_t * (sx2 - sy2)

    sigma = torch.stack(
        [
            torch.stack([sigma_xx, sigma_xy], dim=-1),
            torch.stack([sigma_xy, sigma_yy], dim=-1),
        ],
        dim=-2,
    )
    return xy, sigma


def _gaussian_params_to_mu_sigma_for_loss(params):
    """Convert GauCho [cx, cy, cov_a, cov_b, cov_c] to mu and Sigma."""
    mu = params[..., :2]
    cov_a = params[..., 2]
    cov_b = params[..., 3]
    cov_c = params[..., 4]
    sigma = torch.stack(
        [
            torch.stack([cov_a, cov_c], dim=-1),
            torch.stack([cov_c, cov_b], dim=-1),
        ],
        dim=-2,
    )
    return mu, sigma


def _obb_angle_vector(theta):
    """Return the pi-periodic orientation vector [cos(2theta), sin(2theta)]."""
    return torch.stack(
        [torch.cos(2.0 * theta), torch.sin(2.0 * theta)],
        dim=-1,
    )


@ROTATED_LOSSES.register_module()
class SqrtGaussianProbIoULoss(nn.Module):
    """
    GauCho Gaussian ProbIoU loss with only the high-IoU sqrt refinement.

    Loss:
        L = sqrt(1 - BC)

    This experiment deliberately removes the auxiliary corner and angle
    consistency terms so that the effect of the sqrt-ProbIoU gradient
    can be isolated.

    Expected input from the GauCho head:
        pred_decode    = [cx, cy, cov_a, cov_b, cov_c]
        targets_decode = [cx, cy, w, h, angle]
    """

    def __init__(
            self,
            reduction='mean',
            loss_weight=1.0,
            eps=1e-7):
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(
                "reduction must be one of 'none', 'mean', 'sum'.")
        self.reduction = reduction
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)

    @staticmethod
    def _reduce(loss, reduction, avg_factor=None):
        if reduction == 'none':
            return loss
        if reduction == 'sum':
            return loss.sum()
        if avg_factor is not None:
            return loss.sum() / avg_factor
        return loss.mean()

    def forward(
            self,
            pred,
            target,
            weight=None,
            avg_factor=None,
            reduction_override=None,
            pred_decode=None,
            targets_decode=None,
            **kwargs):
        reduction = (
            reduction_override
            if reduction_override is not None
            else self.reduction
        )
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError('Invalid reduction override.')

        if pred_decode is None or targets_decode is None:
            raise ValueError(
                'SqrtGaussianProbIoULoss requires '
                'pred_decode and targets_decode from GauCho.')

        mu_p, sigma_p = _gaussian_params_to_mu_sigma_for_loss(
            pred_decode)
        mu_t, sigma_t = _xywhr_to_gaussian_sigma_for_loss(
            targets_decode)

        mu_p = mu_p.reshape(-1, 2)
        mu_t = mu_t.reshape(-1, 2)
        sigma_p = sigma_p.reshape(-1, 2, 2)
        sigma_t = sigma_t.reshape(-1, 2, 2)

        eps = self.eps

        det_p = (
            sigma_p[:, 0, 0] * sigma_p[:, 1, 1]
            - sigma_p[:, 0, 1].square()
        ).clamp(min=eps)
        det_t = (
            sigma_t[:, 0, 0] * sigma_t[:, 1, 1]
            - sigma_t[:, 0, 1].square()
        ).clamp(min=eps)

        sigma_avg = 0.5 * (sigma_p + sigma_t)
        a = sigma_avg[:, 0, 0]
        b = sigma_avg[:, 1, 1]
        c = sigma_avg[:, 0, 1]
        det_avg = (a * b - c.square()).clamp(min=eps)

        diff = mu_p - mu_t
        dx = diff[:, 0]
        dy = diff[:, 1]

        bd1 = (
            0.125
            * (a * dy.square() + b * dx.square() - 2.0 * c * dx * dy)
            / det_avg
        )
        bd2 = 0.5 * torch.log(
            det_avg / (torch.sqrt(det_p * det_t) + eps) + eps
        )
        bd = (bd1 + bd2).clamp(min=eps, max=100.0)

        bc = torch.exp(-bd).clamp(max=1.0)

        # High-IoU refinement: sqrt(1 - Bhattacharyya coefficient).
        loss = torch.sqrt(
            (1.0 - bc).clamp(min=eps, max=1.0)
        )

        if weight is not None:
            loss = loss * weight.reshape(-1)

        loss = self._reduce(loss, reduction, avg_factor)
        return self.loss_weight * loss



@ROTATED_LOSSES.register_module()
class IoUAwareQualityLoss(nn.Module):
    """
    Soft IoU-target quality loss for the auxiliary quality branch.

    Target:
        q* = IoU(OBB_pred, OBB_gt)

    The target is detached from the geometry branch. The quality branch
    predicts one scalar logit per feature location and is optimized with
    BCE-with-logits against the continuous IoU target.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(
                "reduction must be one of 'none', 'mean', 'sum'.")
        self.reduction = reduction
        self.loss_weight = float(loss_weight)

    def forward(
            self,
            pred,
            target,
            weight=None,
            avg_factor=None,
            reduction_override=None,
            **kwargs):
        reduction = (
            reduction_override
            if reduction_override is not None
            else self.reduction
        )
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError('Invalid reduction override.')

        pred = pred.reshape(-1)
        target = target.reshape(-1).detach().clamp(0.0, 1.0)

        loss = F.binary_cross_entropy_with_logits(
            pred,
            target,
            reduction='none'
        )

        if weight is not None:
            loss = loss * weight.reshape(-1)

        if reduction == 'none':
            return self.loss_weight * loss
        if reduction == 'sum':
            return self.loss_weight * loss.sum()
        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        else:
            loss = loss.mean()
        return self.loss_weight * loss

# ==============================================================
# Proposed Head
# ==============================================================

@ROTATED_HEADS.register_module()
class ClutterAwareDecoupledGauchoQualityHead(
        RotatedFCOSHead):
    """
    Clutter-Aware Geometry–Uncertainty Decoupled GauCho.

    Geometry branch:
        GauCho Cholesky
            ↓
        Sigma_g
            ↓
        ProbIoU

    Clutter branch:
        Cholesky uncertainty
            ↓
        Sigma_c

    Coupled uncertainty:
        Sigma_total = Sigma_g + lambda_c * Sigma_c

    Training (full-geometry clutter coupling):
        L_bbox = HighIoU-ProbIoU(geometry Gaussian)

        L_clutter = E[-log p(X_gt | mu_pred, Sigma_total)]

        where X_gt is the Gaussian representation of the complete
        GT rotated box. Therefore the clutter-aware NLL supervises
        center + scale + orientation rather than center only.

        No clutter-aware Focus weighting is used in this experiment.

    Inference:
        only geometry GauCho is used.

    Therefore:
        no additional NMS branch
        no additional inference output
    """

    def __init__(
            self,

            gaucho_encoding=False,

            kfiou_loss=False,

            gaussian_centerness=False,

            # ==================================================
            # Proposed geometry-clutter decoupling
            # ==================================================
            decoupled_gaucho=False,

            # Clutter Gaussian:
            # [a_c, b_c, c_c]
            clutter_hidden=None,

            # Cholesky initial sigma in feature coordinates.
            clutter_init_sigma=0.25,

            # Sigma_total =
            # Sigma_geometry +
            # clutter_cov_scale * Sigma_clutter
            clutter_cov_scale=0.5,

            # NLL contribution
            clutter_nll_weight=0.05,

            # Adaptive ProbIoU weighting
            clutter_focus_weight=0.0,

            # Detach clutter branch from regression feature.
            clutter_detach_feature=True,

            # Allow NLL to refine the GauCho geometry branch.
            clutter_nll_detach_geometry=False,

            # Full-geometry coupling mode.
            #   full: center + covariance(wh/theta)
            clutter_geometry_mode='full',

            # High-IoU geometry refinement switch.
            high_iou_geometry=True,

            # IoU-aware localization quality branch.
            iou_quality=True,
            iou_quality_loss_weight=1.0,
            iou_quality_detach_feature=False,
            iou_quality_power=1.0,

            **kwargs
    ):

        super().__init__(**kwargs)

        self.gaucho_encoding = bool(
            gaucho_encoding
        )

        self.kfiou_loss = bool(
            kfiou_loss
        )

        self.gaussian_centerness = bool(
            gaussian_centerness
        )

        self.decoupled_gaucho = bool(
            decoupled_gaucho
        )

        self.clutter_init_sigma = float(
            clutter_init_sigma
        )

        self.clutter_cov_scale = float(
            clutter_cov_scale
        )

        self.clutter_nll_weight = float(
            clutter_nll_weight
        )
        # Zero-weight ablation: instantiate no clutter trainable parameters.
        self.clutter_enabled = self.clutter_nll_weight > 0.0

        self.clutter_focus_weight = float(
            clutter_focus_weight
        )

        self.clutter_detach_feature = bool(
            clutter_detach_feature
        )

        self.clutter_nll_detach_geometry = bool(
            clutter_nll_detach_geometry
        )

        self.clutter_geometry_mode = str(
            clutter_geometry_mode
        )

        self.high_iou_geometry = bool(
            high_iou_geometry
        )

        self.iou_quality = bool(iou_quality)
        self.iou_quality_loss_weight = float(iou_quality_loss_weight)
        self.iou_quality_detach_feature = bool(iou_quality_detach_feature)
        self.iou_quality_power = float(iou_quality_power)

        if self.iou_quality_power <= 0.0:
            raise ValueError('iou_quality_power must be > 0.')

        self.loss_iou_quality = build_loss(
            dict(
                type='IoUAwareQualityLoss',
                reduction='mean',
                loss_weight=1.0
            )
        )

        if self.clutter_geometry_mode != 'full':
            raise ValueError(
                "clutter_geometry_mode must be 'full'."
            )

        self.angle_version = (
            self.bbox_coder.angle_version
        )

        if not self.gaucho_encoding:
            raise ValueError(
                'ClutterAwareDecoupledGauchoQualityHead '
                'requires gaucho_encoding=True.'
            )

        if not self.decoupled_gaucho:
            raise ValueError(
                'decoupled_gaucho must be True.'
            )

        if self.norm_on_bbox:
            raise NotImplementedError(
                'The proposed GauCho implementation '
                'requires norm_on_bbox=False.'
            )

        # ==================================================
        # Clutter Cholesky branch
        #
        # raw:
        #   raw_a
        #   raw_b
        #   raw_rho
        #
        # Convert:
        #   a = softplus(raw_a)
        #   b = softplus(raw_b)
        #
        #   rho = tanh(raw_rho)
        #
        #   c = b*rho/sqrt(1-rho^2)
        #
        # Therefore:
        #
        #   Lc = [[a, 0],
        #         [c, b]]
        #
        # is always positive definite.
        # ==================================================

        if self.clutter_enabled:
            self.conv_clutter = nn.Conv2d(
                self.feat_channels,
                3,
                kernel_size=1,
                padding=0
            )
        else:
            self.conv_clutter = None

        self.conv_quality = nn.Conv2d(
            self.feat_channels,
            1,
            kernel_size=3,
            padding=1
        )

        nn.init.normal_(
            self.conv_quality.weight,
            mean=0.0,
            std=0.01
        )
        nn.init.constant_(self.conv_quality.bias, 0.0)

        # --------------------------------------------------
        # Initialize small clutter uncertainty.
        # This keeps the model close to baseline GauCho
        # at the beginning.
        # --------------------------------------------------

        if self.clutter_enabled:
            init_raw = math.log(
                math.expm1(
                    self.clutter_init_sigma
                )
            )

            nn.init.normal_(
                self.conv_clutter.weight,
                mean=0.0,
                std=1e-3
            )

            nn.init.constant_(
                self.conv_clutter.bias[0],
                init_raw
            )

            nn.init.constant_(
                self.conv_clutter.bias[1],
                init_raw
            )

            nn.init.constant_(
                self.conv_clutter.bias[2],
                0.0
            )

    # ==========================================================
    # Clutter Cholesky decoder
    # ==========================================================

    def _decode_clutter_cholesky(
            self,
            raw,
            stride):

        raw_a = raw[:, 0:1]
        raw_b = raw[:, 1:2]
        raw_rho = raw[:, 2:3]

        a = (
            F.softplus(raw_a)
            + 1e-4
        )

        b = (
            F.softplus(raw_b)
            + 1e-4
        )

        rho = (
            0.95
            * torch.tanh(raw_rho)
        )

        rho_sq = rho.square().clamp(
            max=0.95 ** 2
        )

        c = (
            b
            * rho
            / torch.sqrt(
                1.0 - rho_sq + 1e-6
            )
        )

        # Convert from feature coordinates
        # to image coordinates.
        a = a * stride
        b = b * stride
        c = c * stride

        return torch.cat(
            [a, b, c],
            dim=1
        )

    # ==========================================================
    # Forward
    # ==========================================================

    def forward(self, feats):
        return multi_apply(
            self.forward_single,
            feats,
            self.scales,
            self.strides
        )

    def forward_single(
            self,
            x,
            scale,
            stride):

        cls_score, bbox_pred, cls_feat, reg_feat = \
            super(
                RotatedFCOSHead,
                self
            ).forward_single(x)

        if self.centerness_on_reg:
            centerness = self.conv_centerness(
                reg_feat
            )
        else:
            centerness = self.conv_centerness(
                cls_feat
            )

        bbox_pred = scale(
            bbox_pred
        ).float()

        # --------------------------------------------------
        # Geometry GauCho branch
        # --------------------------------------------------

        gauss_xy = (
            bbox_pred[:, 0:2]
            * stride
        )

        gaucho_ab = (
            bbox_pred[:, 2:4]
            .exp()
            * stride
        )

        gaucho_c = (
            self.conv_angle(reg_feat)
            * stride
        )

        gaucho_bbox_pred = torch.cat(
            [
                gauss_xy,
                gaucho_ab,
                gaucho_c
            ],
            dim=1
        )

        # --------------------------------------------------
        # Clutter Gaussian branch
        # --------------------------------------------------

        if self.clutter_enabled:
            clutter_feat = reg_feat

            if self.clutter_detach_feature:
                clutter_feat = clutter_feat.detach()

            clutter_raw = self.conv_clutter(
                clutter_feat
            )

            clutter_cholesky = (
                self._decode_clutter_cholesky(
                    clutter_raw,
                    stride
                )
            )

        else:
            # Preserve the 5-output interface, but no CAFGUM parameters,
            # convolution, or covariance decoding runs.
            clutter_cholesky = gaucho_bbox_pred.new_zeros(
                (gaucho_bbox_pred.size(0), 3,
                 gaucho_bbox_pred.size(2), gaucho_bbox_pred.size(3)))

        # --------------------------------------------------
        # IoU-aware localization quality branch
        # --------------------------------------------------
        quality_feat = reg_feat
        if self.iou_quality_detach_feature:
            quality_feat = quality_feat.detach()

        quality_score = self.conv_quality(quality_feat)

        # Five outputs are consumed by the custom loss/inference paths:
        # cls, GauCho geometry, clutter covariance, centerness, quality.
        return (
            cls_score,
            gaucho_bbox_pred,
            clutter_cholesky,
            centerness,
            quality_score
        )

    # ==========================================================
    # Loss
    # ==========================================================

    @force_fp32(
        apply_to=(
            'cls_scores',
            'bbox_preds',
            'angle_preds',
            'centernesses',
            'quality_scores'
        )
    )
    def loss(
            self,
            cls_scores,
            bbox_preds,
            angle_preds,
            centernesses,
            quality_scores,
            gt_bboxes,
            gt_labels,
            img_metas,
            gt_bboxes_ignore=None):

        assert (
            len(cls_scores)
            == len(bbox_preds)
            == len(angle_preds)
            == len(centernesses)
            == len(quality_scores)
        )

        featmap_sizes = [
            featmap.size()[-2:]
            for featmap in cls_scores
        ]

        all_level_points = (
            self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=bbox_preds[0].dtype,
                device=bbox_preds[0].device
            )
        )

        labels, bbox_targets, angle_targets = (
            self.get_targets(
                all_level_points,
                gt_bboxes,
                gt_labels
            )
        )

        num_imgs = cls_scores[0].size(0)

        # ==================================================
        # Flatten
        # ==================================================

        flatten_cls_scores = [
            cls_score.permute(
                0, 2, 3, 1
            ).reshape(
                -1,
                self.cls_out_channels
            )
            for cls_score in cls_scores
        ]

        flatten_bbox_preds = [
            bbox_pred.permute(
                0, 2, 3, 1
            ).reshape(
                -1,
                5
            )
            for bbox_pred in bbox_preds
        ]

        flatten_clutter_preds = (
            [clutter_pred.permute(0, 2, 3, 1).reshape(-1, 3)
             for clutter_pred in angle_preds]
            if self.clutter_enabled else None
        )

        flatten_centerness = [
            centerness.permute(
                0, 2, 3, 1
            ).reshape(-1)
            for centerness in centernesses
        ]

        flatten_quality_scores = [
            quality_score.permute(
                0, 2, 3, 1
            ).reshape(-1)
            for quality_score in quality_scores
        ]

        flatten_cls_scores = torch.cat(
            flatten_cls_scores,
            dim=0
        )

        flatten_bbox_preds = torch.cat(
            flatten_bbox_preds,
            dim=0
        )

        if self.clutter_enabled:
            flatten_clutter_preds = torch.cat(flatten_clutter_preds, dim=0)

        flatten_centerness = torch.cat(
            flatten_centerness,
            dim=0
        )

        flatten_quality_scores = torch.cat(
            flatten_quality_scores,
            dim=0
        )

        flatten_labels = torch.cat(
            labels,
            dim=0
        )

        flatten_bbox_targets = torch.cat(
            bbox_targets,
            dim=0
        )

        flatten_angle_targets = torch.cat(
            angle_targets,
            dim=0
        )

        flatten_points = torch.cat(
            [
                points.repeat(
                    num_imgs,
                    1
                )
                for points in all_level_points
            ],
            dim=0
        )

        # ==================================================
        # Classification
        # ==================================================

        bg_class_ind = self.num_classes

        pos_inds = (
            (flatten_labels >= 0)
            &
            (flatten_labels < bg_class_ind)
        ).nonzero().reshape(-1)

        num_pos = torch.tensor(
            len(pos_inds),
            dtype=torch.float,
            device=bbox_preds[0].device
        )

        num_pos = max(
            reduce_mean(num_pos),
            1.0
        )

        loss_cls = self.loss_cls(
            flatten_cls_scores,
            flatten_labels,
            avg_factor=num_pos
        )

        # ==================================================
        # Positive samples
        # ==================================================

        pos_bbox_preds = (
            flatten_bbox_preds[pos_inds]
        )

        pos_clutter_preds = (
            flatten_clutter_preds[pos_inds] if self.clutter_enabled else None
        )

        pos_centerness = (
            flatten_centerness[pos_inds]
        )

        pos_quality_scores = (
            flatten_quality_scores[pos_inds]
        )

        pos_bbox_targets = (
            flatten_bbox_targets[pos_inds]
        )

        pos_angle_targets = (
            flatten_angle_targets[pos_inds]
        )

        zero_loss = (
            flatten_bbox_preds.sum()
            * 0.0
        )

        loss_bbox = zero_loss
        loss_centerness = zero_loss
        loss_clutter_nll = zero_loss
        loss_iou_quality = zero_loss

        if len(pos_inds) > 0:

            pos_points = (
                flatten_points[pos_inds]
            )

            # --------------------------------------------------
            # Build native GT OBB
            # --------------------------------------------------

            pos_target_full = torch.cat(
                [
                    pos_bbox_targets,
                    pos_angle_targets
                ],
                dim=-1
            )

            pos_decoded_target = (
                self.bbox_coder.decode(
                    pos_points,
                    pos_target_full
                )
            )

            # --------------------------------------------------
            # Full GT Gaussian geometry
            # --------------------------------------------------
            if self.clutter_enabled:
                gt_gauss_mean, sigma_gt = xywhr_to_gaussian_sigma(
                    pos_decoded_target.detach()
                )

            # --------------------------------------------------
            # Standard FCOS centerness
            # --------------------------------------------------

            pos_centerness_targets = (
                self.centerness_target(
                    pos_bbox_targets
                )
            )

            # --------------------------------------------------
            # Geometry Gaussian
            # --------------------------------------------------

            pos_pred_gauss = (
                decode_gaucho_offset(
                    pos_points,
                    pos_bbox_preds,
                    self.angle_version,
                    to_obb=False
                )
            )

            pos_pred_obb = (
                decode_gaucho_offset(
                    pos_points,
                    pos_bbox_preds,
                    self.angle_version,
                    to_obb=True
                )
            )

            # --------------------------------------------------
            # IoU-aware localization quality target
            # --------------------------------------------------
            if self.iou_quality:
                with torch.no_grad():
                    aligned_iou = box_iou_rotated(
                        pos_pred_obb.detach(),
                        pos_decoded_target.detach(),
                        mode='iou',
                        aligned=True
                    ).clamp(0.0, 1.0)

                    if self.iou_quality_power != 1.0:
                        quality_target = aligned_iou.pow(
                            self.iou_quality_power
                        )
                    else:
                        quality_target = aligned_iou

                quality_denorm = max(
                    reduce_mean(
                        pos_centerness_targets.sum().detach()
                    ),
                    1e-6
                )

                loss_iou_quality = self.loss_iou_quality(
                    pos_quality_scores,
                    quality_target,
                    weight=pos_centerness_targets,
                    avg_factor=quality_denorm
                )
                loss_iou_quality = (
                    self.iou_quality_loss_weight
                    * loss_iou_quality
                )

            if self.clutter_enabled:
                # --------------------------------------------------
                # Geometry covariance Sigma_g
                # --------------------------------------------------

                geom_cov_a = (
                    pos_pred_gauss[:, 2]
                    .clamp(min=EPS)
                )

                geom_cov_b = (
                    pos_pred_gauss[:, 3]
                    .clamp(min=EPS)
                )

                geom_cov_c = (
                    pos_pred_gauss[:, 4]
                )

                sigma_g = covariance_to_matrix(
                    geom_cov_a,
                    geom_cov_b,
                    geom_cov_c
                )

                # --------------------------------------------------
                # Clutter covariance Sigma_c
                # --------------------------------------------------

                clutter_cov_a, clutter_cov_b, clutter_cov_c = (
                    cholesky_to_covariance(
                        pos_clutter_preds
                    )
                )

                clutter_cov_a = (
                    clutter_cov_a
                    .clamp(min=EPS)
                )

                clutter_cov_b = (
                    clutter_cov_b
                    .clamp(min=EPS)
                )

                sigma_c = covariance_to_matrix(
                    clutter_cov_a,
                    clutter_cov_b,
                    clutter_cov_c
                )

                # --------------------------------------------------
                # Full geometry coupling
                #
                # Sigma_total contains both the predicted GauCho geometry
                # and clutter-induced uncertainty. The NLL now sees the
                # complete GT Gaussian (center + scale + orientation).
                # --------------------------------------------------

                pred_gauss_mean = pos_pred_gauss[:, :2]

                if self.clutter_nll_detach_geometry:
                    sigma_g_nll = sigma_g.detach()
                    pred_mean_nll = pred_gauss_mean.detach()
                else:
                    sigma_g_nll = sigma_g
                    pred_mean_nll = pred_gauss_mean

                sigma_total = (
                    sigma_g_nll
                    + self.clutter_cov_scale * sigma_c
                )

            # --------------------------------------------------
            # Standard FCOS regression weight
            #
            # Experiment 4 is NLL-only:
            #   - ProbIoU uses the original FCOS centerness weight.
            #   - No clutter Focus / adaptive re-weighting is used.
            # --------------------------------------------------

            regression_weight = pos_centerness_targets

            centerness_denorm = max(
                reduce_mean(
                    regression_weight.sum().detach()
                ),
                1e-6
            )

            # --------------------------------------------------
            # Geometry regression
            # --------------------------------------------------

            if self.kfiou_loss:

                pos_pred_xy_offset = (
                    pos_pred_gauss[:, :2]
                    - pos_points
                )

                pos_target_xy_offset = (
                    pos_decoded_target[:, :2]
                    - pos_points
                )

                loss_bbox = self.loss_bbox(
                    pos_pred_xy_offset,
                    pos_target_xy_offset,
                    pred_decode=pos_pred_gauss,
                    targets_decode=pos_decoded_target,
                    weight=regression_weight,
                    avg_factor=centerness_denorm
                )

            else:

                if self.high_iou_geometry:
                    loss_bbox = self.loss_bbox(
                        pos_pred_gauss,
                        pos_decoded_target,
                        weight=regression_weight,
                        avg_factor=centerness_denorm,
                        pred_decode=pos_pred_gauss,
                        targets_decode=pos_decoded_target
                    )
                else:
                    loss_bbox = self.loss_bbox(
                        pos_pred_gauss,
                        pos_decoded_target,
                        weight=regression_weight,
                        avg_factor=centerness_denorm
                    )

            if self.clutter_enabled:
                # ==================================================
                # 2. Full-geometry clutter-aware Gaussian NLL
                # ==================================================

                loss_clutter_nll = gaussian_geometry_nll_2d(
                    pred_mean=pred_mean_nll,
                    pred_sigma=sigma_total,
                    target_mean=gt_gauss_mean,
                    target_sigma=sigma_gt,
                    weight=pos_centerness_targets,
                    avg_factor=max(
                        reduce_mean(
                            pos_centerness_targets.sum().detach()
                        ),
                        1e-6
                    )
                )

                loss_clutter_nll = (
                    self.clutter_nll_weight
                    * loss_clutter_nll
                )

            # ==================================================
            # 3. Centerness
            # ==================================================

            loss_centerness = (
                self.loss_centerness(
                    pos_centerness,
                    pos_centerness_targets,
                    avg_factor=num_pos
                )
            )

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_clutter_nll=loss_clutter_nll,
            loss_iou_quality=loss_iou_quality
        )

    # ==========================================================
    # Inference
    # ==========================================================

    @force_fp32(
        apply_to=(
            'cls_scores',
            'bbox_preds',
            'angle_preds',
            'centernesses',
            'quality_scores'
        )
    )
    def get_bboxes(
            self,
            cls_scores,
            bbox_preds,
            angle_preds,
            centernesses,
            quality_scores,
            img_metas,
            cfg=None,
            rescale=None):
        """Decode a batch of predictions with IoU-aware quality scoring."""

        assert (
            len(cls_scores)
            == len(bbox_preds)
            == len(angle_preds)
            == len(centernesses)
            == len(quality_scores)
        )

        num_levels = len(cls_scores)
        featmap_sizes = [
            featmap.size()[-2:]
            for featmap in cls_scores
        ]
        mlvl_points = self.prior_generator.grid_priors(
            featmap_sizes,
            bbox_preds[0].dtype,
            bbox_preds[0].device
        )

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach()
                for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach()
                for i in range(num_levels)
            ]
            angle_pred_list = [
                angle_preds[i][img_id].detach()
                for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach()
                for i in range(num_levels)
            ]
            quality_score_list = [
                quality_scores[i][img_id].detach()
                for i in range(num_levels)
            ]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']

            result_list.append(
                self._get_bboxes_single(
                    cls_score_list,
                    bbox_pred_list,
                    angle_pred_list,
                    centerness_pred_list,
                    quality_score_list,
                    mlvl_points,
                    img_shape,
                    scale_factor,
                    cfg,
                    rescale
                )
            )

        return result_list

    def _get_bboxes_single(
            self,
            cls_scores,
            bbox_preds,
            angle_preds,
            centernesses,
            quality_scores,
            mlvl_points,
            img_shape,
            scale_factor,
            cfg,
            rescale=False):
        """Decode one image and fuse class, centerness and quality scores."""

        cfg = self.test_cfg if cfg is None else cfg
        use_lqc_scoring = cfg.get('lqc_scoring', True)

        assert (
            len(cls_scores)
            == len(bbox_preds)
            == len(angle_preds)
            == len(centernesses)
            == len(quality_scores)
            == len(mlvl_points)
        )

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_score_factors = []

        for (
                cls_score,
                bbox_pred,
                angle_pred,
                centerness,
                quality_score,
                points
        ) in zip(
            cls_scores,
            bbox_preds,
            angle_preds,
            centernesses,
            quality_scores,
            mlvl_points
        ):
            assert (
                cls_score.size()[-2:]
                == bbox_pred.size()[-2:]
            )

            scores = (
                cls_score
                .permute(1, 2, 0)
                .reshape(-1, self.cls_out_channels)
                .sigmoid()
            )
            centerness = (
                centerness
                .permute(1, 2, 0)
                .reshape(-1)
                .sigmoid()
            )
            # quality = (
            #     quality_score
            #     .permute(1, 2, 0)
            #     .reshape(-1)
            #     .sigmoid()
            # )

            bbox_pred = (
                bbox_pred
                .permute(1, 2, 0)
                .reshape(-1, 5)
            )
            # if self.iou_quality:
            #     quality = (
            #         quality_score
            #         .permute(1, 2, 0)
            #         .reshape(-1)
            #         .sigmoid()
            #     )
            #
            #     score_factor = centerness * quality
            # else:
            #     # Experiment B / C / D without IoU Quality
            #     score_factor = centerness
            # # score_factor = centerness * quality
            # LQC inference scoring ablation
            if self.iou_quality and use_lqc_scoring:
                quality = (
                    quality_score
                    .permute(1, 2, 0)
                    .reshape(-1)
                    .sigmoid()
                )
                score_factor = centerness * quality
            else:
                score_factor = centerness

            nms_pre = cfg.get('nms_pre', -1)
            if (
                nms_pre > 0
                and scores.shape[0] > nms_pre
            ):
                max_scores, _ = (
                    scores * score_factor[:, None]
                ).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)

                points = points[topk_inds]
                bbox_pred = bbox_pred[topk_inds]
                scores = scores[topk_inds]
                score_factor = score_factor[topk_inds]

            bboxes = decode_gaucho_offset(
                points,
                bbox_pred,
                self.angle_version,
                to_obb=True
            )

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_score_factors.append(score_factor)

        mlvl_bboxes = torch.cat(mlvl_bboxes)

        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = (
                mlvl_bboxes[..., :4]
                / scale_factor
            )

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(
            mlvl_scores.shape[0],
            1
        )
        mlvl_scores = torch.cat(
            [mlvl_scores, padding],
            dim=1
        )

        mlvl_score_factors = torch.cat(
            mlvl_score_factors
        )

        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_score_factors
        )

        return det_bboxes, det_labels
