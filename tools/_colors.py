"""色彩空间转换工具 — LAB 阈值常量与 MV↔CV 格式互转。"""

import numpy as np

MatLike = np.ndarray

# 矩形检测的默认 LAB 色彩空间阈值（Machine Vision LAB 格式）
RECT_MV_LABVALUE = [7, 32, -13, 9, -13, 21]


def cvt_mvlab2cv(param: np.array = RECT_MV_LABVALUE):
    """将机器视觉 LAB 格式的阈值转换为 OpenCV LAB 格式。"""
    lmin, lmax, amin, amax, bmin, bmax = param
    lower_bound = np.array([int(lmin * 2.55), int(amin + 128), int(bmin + 128)])
    upper_bound = np.array([int(lmax * 2.55), int(amax + 128), int(bmax + 128)])
    return lower_bound, upper_bound
