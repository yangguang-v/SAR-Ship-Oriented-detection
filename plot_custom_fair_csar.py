
"""Plot CholSAR (FAIR-CSAR-v1.0, epoch 31) rotated confusion matrix.

Run at the MMRotate project root, after generating predictions_epoch31.pkl:
    python plot_cholsar_confusion_epoch31.py

Uses the project's official rotated IoU confusion-matrix implementation.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 服务器无图形桌面时也能保存图像
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import seaborn as sns
from mmcv import Config
from mmrotate.datasets import build_dataset

# 在 MMRotate 项目根目录执行；使用官方旋转框混淆矩阵实现（不使用普通 HBB IoU）
from tools.analysis_tools.confusion_matrix import calculate_confusion_matrix

# ========== 配置 ==========
CONFIG_FILE = 'configs_gaucho/mydetect/gaussian_fcos_lsk_fpn_clutter_iou_quality_fair_csar_E.py'
WORK_DIR = Path('work_dirs/gaussian_fcos_lsk_fpn_clutter_iou_quality_fair_csar_fixed_E')
PKL_FILE = WORK_DIR / 'predictions_epoch31.pkl'
OUTPUT_DIR = WORK_DIR / 'confusion_matrix_epoch31'
SCORE_THR = 0.1
IOU_THR = 0.5

# 论文作图类别顺序；背景置于最后
ORDER_WITH_BG = ['Buc', 'Cos', 'Gcs', 'Mds', 'Oit', 'Ots', 'Was', 'background']


def main():
    if not Path(CONFIG_FILE).is_file():
        raise FileNotFoundError(f'模型配置不存在: {CONFIG_FILE}')
    if not PKL_FILE.is_file():
        raise FileNotFoundError(
            f'缺少预测文件: {PKL_FILE}\n'
            '先运行 tools/test.py，并使用 --out 保存 predictions_epoch31.pkl。')

    cfg = Config.fromfile(CONFIG_FILE)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(str(PKL_FILE))

    # MMRotate 的常见格式: results[image_idx][class_idx] = ndarray(N,6)
    if not isinstance(results, list):
        raise TypeError(f'预测结果必须为 list，实际为 {type(results).__name__}')
    if len(results) != len(dataset):
        raise ValueError(
            f'预测文件中有 {len(results)} 张图，而 cfg.data.test 有 {len(dataset)} 张图。'
            '请确保生成 PKL 和绘图使用相同的数据集、图片顺序及配置。')
    if results and isinstance(results[0], tuple):
        results = [item[0] for item in results]
    if results and not isinstance(results[0], list):
        raise TypeError(f'单张图预测结果必须为按类别排列的 list，实际为 {type(results[0]).__name__}')

    actual_classes = list(dataset.CLASSES) + ['background']
    if set(actual_classes) != set(ORDER_WITH_BG) or len(actual_classes) != len(ORDER_WITH_BG):
        raise ValueError(
            f'数据集类别与预期不匹配: {actual_classes}\n'
            f'绘图类别: {ORDER_WITH_BG}')

    print(f'正在计算 CholSAR epoch31 混淆矩阵，图像数: {len(dataset)}')
    print(f'阈值: score >= {SCORE_THR}, rotated IoU >= {IOU_THR}')
    matrix = calculate_confusion_matrix(
        dataset, results,
        score_thr=SCORE_THR,
        nms_iou_thr=None,
        tp_iou_thr=IOU_THR,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_DIR / 'confusion_matrix.npy'), matrix)

    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(
        matrix, row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=(row_sums != 0),
    )
    idx_order = [actual_classes.index(name) for name in ORDER_WITH_BG]
    norm_matrix = norm_matrix[np.ix_(idx_order, idx_order)]

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        norm_matrix, annot=True, fmt='.2%', cmap='Blues', square=True,
        xticklabels=ORDER_WITH_BG, yticklabels=ORDER_WITH_BG,
        linewidths=0.5, cbar_kws={'shrink': 0.8},
        annot_kws={'size': 10, 'weight': 'bold'}, ax=ax,
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('bottom')
    ax.tick_params(axis='x', labelrotation=45, labelsize=13)
    ax.tick_params(axis='y', labelrotation=0, labelsize=13)
    ax.set_xlabel('Prediction Label', fontsize=15, fontweight='bold')
    ax.set_ylabel('Ground Truth Label', fontsize=15, fontweight='bold')
    ax.set_title('CholSAR (Epoch 31) - Normalized Confusion Matrix',
                 fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    pdf_path = OUTPUT_DIR / 'confusion_matrix_epoch31.pdf'
    png_path = OUTPUT_DIR / 'confusion_matrix_epoch31.png'
    fig.savefig(str(pdf_path), dpi=300, bbox_inches='tight')
    fig.savefig(str(png_path), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'混淆矩阵: {OUTPUT_DIR / "confusion_matrix.npy"}')
    print(f'PDF: {pdf_path}')
    print(f'PNG: {png_path}')


if __name__ == '__main__':
    main()

