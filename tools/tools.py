"""工具模块 — 向后兼容的重导出入口。

原 monolithic tools.py 已拆分为以下子模块:
    _colors       — MatLike, RECT_MV_LABVALUE, cvt_mvlab2cv
    _geometry     — order_points, perspective_correct_and_validate
    _drawing      — DrawGraph
    _threshold    — Binarizer, AutoThresholder (@deprecated)
    _rect_detect  — _process_contour_chunk, detect_rect
    _laser_detect — LaserSpot, detect_laser_mask, detect_laser_binary
    _tracking     — FpsShow, LaserSpotDetector, RectTracker
    _preprocess   — preprocess

所有公开 API 通过本文件重新导出，现有 `from tools.tools import ...` 无需修改。
"""

from tools._colors import MatLike, RECT_MV_LABVALUE, cvt_mvlab2cv
from tools._geometry import order_points, perspective_correct_and_validate
from tools._drawing import DrawGraph
from tools._threshold import AutoThresholder, Binarizer
from tools._rect_detect import _process_contour_chunk, detect_rect
from tools._laser_detect import LaserSpot, detect_laser_binary, detect_laser_mask
from tools._tracking import FpsShow, LaserSpotDetector, RectTracker
from tools._preprocess import preprocess

__all__ = [
    # _colors
    "MatLike",
    "RECT_MV_LABVALUE",
    "cvt_mvlab2cv",
    # _geometry
    "order_points",
    "perspective_correct_and_validate",
    # _drawing
    "DrawGraph",
    # _threshold
    "AutoThresholder",  # @deprecated — 请迁移到 Binarizer
    "Binarizer",
    # _rect_detect
    "_process_contour_chunk",
    "detect_rect",
    # _laser_detect
    "LaserSpot",
    "detect_laser_binary",
    "detect_laser_mask",
    # _tracking
    "FpsShow",
    "LaserSpotDetector",
    "RectTracker",
    # _preprocess
    "preprocess",
]
