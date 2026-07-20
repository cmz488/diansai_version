"""追踪器 — FPS 显示、激光点追踪与矩形 ROI 追踪。"""

import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ._colors import cvt_mvlab2cv
from ._laser_detect import LaserSpot
from ._rect_detect import detect_rect

MatLike = np.ndarray


# ============================================================================
# FPS 显示
# ============================================================================

class FpsShow:
    """实时 FPS 显示工具。"""

    def __init__(self) -> None:
        self.last_ = time.time()
        self.fps = 0.0

    def show(self, frame: MatLike) -> MatLike:
        current = time.time()
        self.fps = 1 / (current - self.last_)
        self.last_ = current
        h = frame.shape[0]
        w = frame.shape[1]
        frame = cv2.putText(
            frame,
            "fps:{:.2f}".format(self.fps),
            (int(w * 0.8), int(h * 0.2)),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (0, 255, 0),
        )
        return frame


# ============================================================================
# 激光点追踪器
# ============================================================================

class LaserSpotDetector:
    """激光点检测器 — LAB 红色掩码法 + ROI 追踪，集成中心平滑。"""

    def __init__(
        self,
        track_radius: int = 120,
        smooth_alpha: float = 0.65,
        full_search_interval: int = 30,
        min_area: int = 10,
    ) -> None:
        if not 0.0 < smooth_alpha <= 1.0:
            raise ValueError("smooth_alpha 必须在 (0, 1] 范围内")

        self.track_radius = track_radius
        self.smooth_alpha = smooth_alpha
        self.full_search_interval = full_search_interval
        self.min_area = min_area

        self._last_position: Optional[Tuple[float, float]] = None
        self._smoothed_position: Optional[Tuple[float, float]] = None
        self._frame_count: int = 0
        self._track_hits: int = 0
        self._full_searches: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def detect(
        self, frame: MatLike,
    ) -> Optional[LaserSpot]:
        self._frame_count += 1
        spot: Optional[LaserSpot] = None

        use_tracking = self._last_position is not None and (
            self.full_search_interval <= 0
            or self._frame_count % self.full_search_interval != 0
        )

        if use_tracking:
            spot = self._detect_tracking(frame)
            if spot is not None:
                self._track_hits += 1
            else:
                self._misses += 1
                spot = self._detect_full(frame)
        else:
            self._full_searches += 1
            spot = self._detect_full(frame)

        if spot is not None:
            self._last_position = (spot.x, spot.y)
            sx, sy = self._apply_smoothing(spot.x, spot.y)
            return LaserSpot(
                x=sx, y=sy,
                peak_value=spot.peak_value,
                confidence=spot.confidence,
                noise_level=spot.noise_level,
                search_mode=spot.search_mode,
            )

        if self._misses > 10:
            self._last_position = None
            self._smoothed_position = None

        return None

    def reset(self) -> None:
        self._last_position = None
        self._smoothed_position = None
        self._frame_count = 0
        self._track_hits = 0
        self._full_searches = 0
        self._misses = 0

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_frames": self._frame_count,
            "track_hits": self._track_hits,
            "full_searches": self._full_searches,
            "misses": self._misses,
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _detect_tracking(
        self, frame: MatLike,
    ) -> Optional[LaserSpot]:
        """ROI 追踪 — 使用红色掩码法在上一帧位置附近检测激光点。"""
        assert self._last_position is not None
        lx, ly = self._last_position
        h, w = frame.shape[:2]
        r = self.track_radius

        x1 = max(0, int(lx) - r)
        x2 = min(w, int(lx) + r)
        y1 = max(0, int(ly) - r)
        y2 = min(h, int(ly) + r)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        result = self._detect_by_mask(lab_roi, gray_roi)
        if result is None:
            return None

        cx_roi, cy_roi, peak_value = result
        cx = cx_roi + x1
        cy = cy_roi + y1

        return self._make_spot(cx, cy, peak_value, "tracking")

    def _detect_full(
        self, frame: MatLike,
    ) -> Optional[LaserSpot]:
        """全图搜索 — 使用红色掩码法检测激光点。"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = self._detect_by_mask(lab, gray)
        if result is None:
            return None
        cx, cy, peak_value = result
        return self._make_spot(cx, cy, peak_value, "full")

    # ------------------------------------------------------------------
    # 检测方法
    # ------------------------------------------------------------------

    def _detect_by_mask(
        self, lab_frame: MatLike, gray_frame: MatLike,
    ) -> Optional[Tuple[float, float, float]]:
        """红色掩码法 — 通过 LAB 红色掩码检测激光点，返回 (cx, cy, peak_value) 或 None。"""
        ls, us = cvt_mvlab2cv([67, 100, 4, 22, -45, -14])

        mask = cv2.inRange(lab_frame, ls, us)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        red_laser = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

        edge = cv2.Canny(red_laser, 50, 150)
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_laser = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best_laser) < self.min_area:
            return None

        x, y, w, h = cv2.boundingRect(best_laser)
        cx = x + w / 2.0
        cy = y + h / 2.0

        if np.any(mask > 0):
            peak_value = float(gray_frame[mask > 0].max())
        else:
            peak_value = 0.0

        return cx, cy, peak_value

    def _make_spot(
        self, x: float, y: float, peak_value: float, search_mode: str
    ) -> LaserSpot:
        confidence = min(peak_value / 255.0, 1.0)
        return LaserSpot(
            x=x, y=y, peak_value=peak_value,
            confidence=confidence, noise_level=0.0,
            search_mode=search_mode,
        )

    def _apply_smoothing(self, x: float, y: float) -> Tuple[float, float]:
        if self._smoothed_position is None:
            self._smoothed_position = (x, y)
            return (x, y)
        sx, sy = self._smoothed_position
        alpha = self.smooth_alpha
        nx = alpha * x + (1.0 - alpha) * sx
        ny = alpha * y + (1.0 - alpha) * sy
        self._smoothed_position = (nx, ny)
        return (nx, ny)


# ============================================================================
# 矩形 ROI 追踪器
# ============================================================================

class RectTracker:
    """矩形 ROI 追踪器 — ROI 加速 + 中心平滑 + 定时全图回退。"""

    def __init__(
        self,
        track_radius: int = 250,
        smooth_alpha: float = 0.6,
        full_search_interval: int = 30,
    ) -> None:
        if not 0.0 < smooth_alpha <= 1.0:
            raise ValueError("smooth_alpha 必须在 (0, 1] 范围内")
        self.track_radius = track_radius
        self.smooth_alpha = smooth_alpha
        self.full_search_interval = full_search_interval
        self._last_center: Optional[Tuple[float, float]] = None
        self._smoothed_center: Optional[Tuple[float, float]] = None
        self._frame_count: int = 0
        self._track_hits: int = 0
        self._full_searches: int = 0
        self._misses: int = 0

    def track(
        self,
        edges: MatLike,
        gray: MatLike,
        min_area: np.uint32,
        white_area: np.uint32,
        real_aspect_ratio: float,
        target_width: Optional[int] = None,
        tolerance: float = 0.1,
        epsilon: float = 0.02,
        reject_status: Optional[Dict] = None,
    ) -> Optional[MatLike]:
        if reject_status is None:
            reject_status = {
                "area": 0, "quad": 0, "white_region": 0, "aspect_ratio": 0,
            }

        self._frame_count += 1

        use_tracking = self._last_center is not None and (
            self.full_search_interval <= 0
            or self._frame_count % self.full_search_interval != 0
        )

        if use_tracking:
            best_rect = self._track_roi(
                edges, gray, min_area, white_area,
                real_aspect_ratio, target_width=target_width,
                tolerance=tolerance, epsilon=epsilon,
                reject_status=reject_status,
            )
            if best_rect is None:
                self._misses += 1
                best_rect = detect_rect(
                    edges, gray, min_area, white_area,
                    real_aspect_ratio, target_width=target_width,
                    tolerance=tolerance, epsilon=epsilon,
                    reject_status=reject_status,
                )
            else:
                self._track_hits += 1
        else:
            best_rect = detect_rect(
                edges, gray, min_area, white_area,
                real_aspect_ratio, target_width=target_width,
                tolerance=tolerance, epsilon=epsilon,
                reject_status=reject_status,
            )
            self._full_searches += 1

        if best_rect is not None:
            cx = float(best_rect[:, 0].mean())
            cy = float(best_rect[:, 1].mean())
            self._last_center = (cx, cy)
            if self._smoothed_center is None:
                self._smoothed_center = (cx, cy)
            else:
                scx, scy = self._smoothed_center
                self._smoothed_center = (
                    self.smooth_alpha * cx + (1.0 - self.smooth_alpha) * scx,
                    self.smooth_alpha * cy + (1.0 - self.smooth_alpha) * scy,
                )
        else:
            self._misses += 1
            if self._misses > 10:
                self._last_center = None
                self._smoothed_center = None

        return best_rect

    def reset(self) -> None:
        self._last_center = None
        self._smoothed_center = None
        self._frame_count = 0
        self._track_hits = 0
        self._full_searches = 0
        self._misses = 0

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_frames": self._frame_count,
            "track_hits": self._track_hits,
            "full_searches": self._full_searches,
            "misses": self._misses,
        }

    @property
    def smoothed_center(self) -> Optional[Tuple[float, float]]:
        return self._smoothed_center

    def _track_roi(
        self,
        edges: MatLike,
        gray: MatLike,
        min_area: np.uint32,
        white_area: np.uint32,
        real_aspect_ratio: float,
        target_width: Optional[int],
        tolerance: float,
        epsilon: float,
        reject_status: Dict,
    ) -> Optional[MatLike]:
        assert self._last_center is not None
        lx, ly = self._last_center
        h, w = edges.shape[:2]
        r = self.track_radius

        x1 = max(0, int(lx) - r)
        x2 = min(w, int(lx) + r)
        y1 = max(0, int(ly) - r)
        y2 = min(h, int(ly) + r)

        if x2 <= x1 or y2 <= y1:
            return None

        roi_edges = edges[y1:y2, x1:x2]
        roi_gray = gray[y1:y2, x1:x2]

        local_rect = detect_rect(
            roi_edges, roi_gray, min_area, white_area,
            real_aspect_ratio, target_width=target_width,
            tolerance=tolerance, epsilon=epsilon,
            reject_status=reject_status, num_workers=1,
        )

        if local_rect is None:
            return None

        local_rect = local_rect.astype(np.int32)
        local_rect[:, 0] += x1
        local_rect[:, 1] += y1
        return local_rect
