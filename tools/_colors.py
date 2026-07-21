"""色彩空间转换工具 — LAB 阈值常量与 MV↔CV 格式互转。"""

import numpy as np

MatLike = np.ndarray

# 矩形检测的默认 LAB 色彩空间阈值（Machine Vision LAB 格式）
RECT_MV_LABVALUE = [7, 32, -13, 9, -13, 21]


def cvt_mvlab2cv(param=RECT_MV_LABVALUE):
    """将机器视觉 LAB 阈值转换为 OpenCV LAB 的 ``uint8`` 阈值。

    机器视觉格式使用 L=[0, 100]、A/B=[-128, 127]。这里不能直接
    ``int(value * 2.55)``，因为浮点截断会把 L=100 错误转换为 254。
    """
    values = np.asarray(param, dtype=np.float64).reshape(-1)
    if values.size != 6 or not np.all(np.isfinite(values)):
        raise ValueError("LAB 阈值必须包含 6 个有限数值")

    lmin, lmax, amin, amax, bmin, bmax = values
    if not (0 <= lmin <= lmax <= 100):
        raise ValueError("L 阈值必须满足 0 <= min <= max <= 100")
    if not (-128 <= amin <= amax <= 127):
        raise ValueError("A 阈值必须满足 -128 <= min <= max <= 127")
    if not (-128 <= bmin <= bmax <= 127):
        raise ValueError("B 阈值必须满足 -128 <= min <= max <= 127")

    lower_bound = np.rint([lmin * 255.0 / 100.0, amin + 128, bmin + 128])
    upper_bound = np.rint([lmax * 255.0 / 100.0, amax + 128, bmax + 128])
    return (
        np.clip(lower_bound, 0, 255).astype(np.uint8),
        np.clip(upper_bound, 0, 255).astype(np.uint8),
    )
