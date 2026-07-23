"""CamShift 颜色直方图追踪器 — 通用帧间 ROI 预测。

不负责精确检测，只负责根据颜色直方图预测目标的大致位置。
调用方在预测的 ROI 内做精确检测（如 geom 角点提取）。

用法::

    # 默认：HSV H+S+V 3D 直方图（完整颜色信息）
    tracker = CamShiftTracker()
    tracker.init(frame_bgr, (x, y, w, h))
    bbox = tracker.predict(frame_bgr)

    # 纯色相（适合饱和度高的目标）
    tracker = CamShiftTracker(color_space="hsv_h")

    # 仅亮度（适合灰度/黑白目标）
    tracker = CamShiftTracker(color_space="lab_l")
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

MatLike = np.ndarray


# ============================================================================
# 颜色空间配置
# ============================================================================

_COLOR_SPACES = {
    "hsv": {
        "convert": cv2.COLOR_BGR2HSV,
        "channels": [0, 1, 2],             # H + S + V
        "hist_size": [30, 32, 32],          # 粗粒度防稀疏
        "ranges": [0, 180, 0, 256, 0, 256],
        "desc": "HSV H+S+V 3D",
    },
    "hsv_hs": {
        "convert": cv2.COLOR_BGR2HSV,
        "channels": [0, 1],                 # H + S
        "hist_size": [180, 32],
        "ranges": [0, 180, 0, 256],
        "desc": "HSV H+S 2D",
    },
    "hsv_h": {
        "convert": cv2.COLOR_BGR2HSV,
        "channels": [0],                    # H only
        "hist_size": [180],
        "ranges": [0, 180],
        "desc": "HSV H 1D",
    },
    "lab_l": {
        "convert": cv2.COLOR_BGR2LAB,
        "channels": [0],                    # L only
        "hist_size": [256],
        "ranges": [0, 256],
        "desc": "LAB L 1D",
    },
}


# ============================================================================
# CamShiftTracker
# ============================================================================


class CamShiftTracker:
    """基于颜色直方图的 CamShift 帧间预测器。

    接口：
        init(frame_bgr, bbox)   — 从 bbox (x,y,w,h) 区域建直方图
        predict(frame_bgr)      — 预测新位置，返回 (x,y,w,h) 或 None
        ready                   — 是否已初始化
        reset()                 — 清除状态

    predict 返回的 bbox 已按 margin 膨胀，可直接裁剪图像后做精确检测。
    连续丢失超过 max_misses 次后自动 reset。
    """

    def __init__(
        self,
        margin: float = 0.3,
        max_misses: int = 10,
        color_space: str = "hsv",
    ) -> None:
        if not 0.1 <= margin <= 1.0:
            raise ValueError("margin 必须在 [0.1, 1.0] 范围内")
        if color_space not in _COLOR_SPACES:
            raise ValueError(
                f"color_space 必须是 {'/'.join(_COLOR_SPACES)}，收到 {color_space}"
            )

        self.margin = margin
        self.max_misses = max_misses
        self.color_space = color_space
        self._cfg = _COLOR_SPACES[color_space]

        self._hist: Optional[MatLike] = None
        self._window: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
        self._miss_count: int = 0

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def init(self, frame_bgr: MatLike, bbox: Tuple[int, int, int, int]) -> None:
        """从 bbox (x, y, w, h) 区域建颜色直方图。

        构建完成后 ready 变为 True，可调用 predict()。
        """
        x, y, w, h = bbox
        h_img, w_img = frame_bgr.shape[:2]
        x = max(0, x)
        y = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x2 <= x or y2 <= y:
            raise ValueError(f"bbox {bbox} 在图像范围外")

        # 3D 直方图需要足够多的采样像素，否则极度稀疏
        pixels = (x2 - x) * (y2 - y)
        total_bins = int(np.prod(self._cfg["hist_size"]))
        if pixels < total_bins * 2:
            print(
                f"[camshift] 警告: bbox 只有 {pixels} 像素，"
                f"但直方图有 {total_bins} bins ({self._cfg['desc']})，"
                f"建议用更大的框选区域，或改用 hsv_h / hsv_hs"
            )

        converted = cv2.cvtColor(frame_bgr, self._cfg["convert"])

        mask = np.zeros(converted.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x2, y2), 255, -1)

        self._hist = cv2.calcHist(
            [converted],
            self._cfg["channels"],
            mask,
            self._cfg["hist_size"],
            self._cfg["ranges"],
        )
        cv2.normalize(self._hist, self._hist, 0, 255, cv2.NORM_MINMAX)
        self._window = (x, y, x2 - x, y2 - y)
        self._miss_count = 0

    def predict(
        self, frame_bgr: MatLike
    ) -> Optional[Tuple[int, int, int, int]]:
        """运行 CamShift，返回膨胀后的搜索区域 (x, y, w, h) 或 None。

        连续丢失超过 max_misses 次时自动 reset，之后 predict 返回 None 直到重新 init。
        """
        if self._hist is None or self._window is None:
            return None

        converted = cv2.cvtColor(frame_bgr, self._cfg["convert"])

        back_proj = cv2.calcBackProject(
            [converted],
            self._cfg["channels"],
            self._hist,
            self._cfg["ranges"],
            1,
        )

        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        try:
            _rot_rect, window = cv2.CamShift(back_proj, self._window, term_crit)
        except cv2.error:
            return self._record_miss()

        # 低质量过滤
        if window[2] < 5 or window[3] < 5:
            return self._record_miss()
        area = window[2] * window[3]
        orig_area = self._window[2] * self._window[3]
        if orig_area > 0 and area < orig_area * 0.05:
            return self._record_miss()

        self._window = window
        self._miss_count = 0

        # 膨胀 margin，返回 (x, y, w, h)
        x, y, w, h = window
        margin_w = int(w * self.margin)
        margin_h = int(h * self.margin)

        h_img, w_img = frame_bgr.shape[:2]
        nx = max(0, x - margin_w)
        ny = max(0, y - margin_h)
        nw = min(w_img, x + w + margin_w) - nx
        nh = min(h_img, y + h + margin_h) - ny
        return (nx, ny, max(nw, 1), max(nh, 1))

    @property
    def ready(self) -> bool:
        return self._hist is not None

    @property
    def miss_count(self) -> int:
        return self._miss_count

    def reset(self) -> None:
        self._hist = None
        self._window = None
        self._miss_count = 0

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _record_miss(self) -> None:
        """记录一次丢失；超过阈值自动 reset。"""
        self._miss_count += 1
        if self._miss_count >= self.max_misses:
            self.reset()
        return None
