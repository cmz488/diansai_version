"""追踪器 — FPS 显示、激光点追踪与矩形 ROI 追踪。"""

import time
from typing import Dict, NamedTuple, Optional, Sequence, Tuple

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


class _LaserCandidate(NamedTuple):
    x: float
    y: float
    peak_value: float
    confidence: float
    noise_level: float
    rank_score: float


class LaserSpotDetector:
    """单帧激光点检测器 — LAB 掩码、连通域评分与 ROI 追踪。"""

    def __init__(
        self,
        track_radius: int = 120,
        smooth_alpha: float = 0.65,
        full_search_interval: int = 30,
        min_area: int = 10,
        threshold: Sequence[float] = (99, 100, -32, 28, -38, 26),
        max_area: int = 1000,
        morph_kernel_size: int = 3,
        roi_margin: int = 4,
        max_aspect_ratio: float = 3.0,
        min_confidence: float = 0.25,
        color_mode: str = "blue",
        min_color_excess: int = 40,
        min_color_value: int = 80,
        max_consecutive_misses: int = 10,
    ) -> None:
        if not 0.0 < smooth_alpha <= 1.0:
            raise ValueError("smooth_alpha 必须在 (0, 1] 范围内")
        if track_radius <= 0:
            raise ValueError("track_radius 必须 > 0")
        if min_area <= 0 or max_area < min_area:
            raise ValueError("面积阈值必须满足 0 < min_area <= max_area")
        if morph_kernel_size < 0 or (
            morph_kernel_size > 1 and morph_kernel_size % 2 == 0
        ):
            raise ValueError("morph_kernel_size 必须为 0、1 或大于 1 的奇数")
        if roi_margin < 0:
            raise ValueError("roi_margin 必须 >= 0")
        if max_aspect_ratio < 1.0:
            raise ValueError("max_aspect_ratio 必须 >= 1")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence 必须在 [0, 1] 范围内")
        if color_mode not in ("blue", "red", "any"):
            raise ValueError("color_mode 必须是 'blue'、'red' 或 'any'")
        if not 0 <= min_color_excess <= 255 or not 0 <= min_color_value <= 255:
            raise ValueError("颜色强度阈值必须在 [0, 255] 范围内")
        if max_consecutive_misses < 1:
            raise ValueError("max_consecutive_misses 必须 >= 1")

        # 在初始化阶段完成阈值合法性验证。
        cvt_mvlab2cv(threshold)

        self.track_radius = track_radius
        self.smooth_alpha = smooth_alpha
        self.full_search_interval = full_search_interval
        self.min_area = min_area
        self.max_area = max_area
        self.morph_kernel_size = morph_kernel_size
        self.roi_margin = roi_margin
        self.max_aspect_ratio = max_aspect_ratio
        self.min_confidence = min_confidence
        self.color_mode = color_mode
        self.min_color_excess = min_color_excess
        self.min_color_value = min_color_value
        self.max_consecutive_misses = max_consecutive_misses
        self.threshold = tuple(float(value) for value in threshold)

        self._last_position: Optional[Tuple[float, float]] = None
        self._smoothed_position: Optional[Tuple[float, float]] = None
        self._frame_count: int = 0
        self._track_hits: int = 0
        self._full_searches: int = 0
        self._total_misses: int = 0
        self._consecutive_misses: int = 0

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: MatLike,
        search_polygon: Optional[MatLike] = None,
    ) -> Optional[LaserSpot]:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame 必须是 H×W×3 的 BGR 图像")

        self._frame_count += 1
        spot: Optional[LaserSpot] = None
        search_mask = self._build_search_mask(frame.shape[:2], search_polygon)
        if search_mask is not None and cv2.countNonZero(search_mask) == 0:
            self._record_miss()
            return None

        use_tracking = self._last_position is not None and (
            self.full_search_interval <= 0
            or self._frame_count % self.full_search_interval != 0
        )

        if use_tracking:
            spot = self._detect_tracking(frame, search_mask)
            if spot is not None:
                self._track_hits += 1
            else:
                self._full_searches += 1
                spot = self._detect_full(frame, search_mask)
        else:
            self._full_searches += 1
            spot = self._detect_full(frame, search_mask)

        if spot is not None:
            self._consecutive_misses = 0
            self._last_position = (spot.x, spot.y)
            sx, sy = self._apply_smoothing(spot.x, spot.y)
            return LaserSpot(
                x=sx,
                y=sy,
                peak_value=spot.peak_value,
                confidence=spot.confidence,
                noise_level=spot.noise_level,
                search_mode=spot.search_mode,
            )

        self._record_miss()
        return None

    def reset(self) -> None:
        self._last_position = None
        self._smoothed_position = None
        self._frame_count = 0
        self._track_hits = 0
        self._full_searches = 0
        self._total_misses = 0
        self._consecutive_misses = 0

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_frames": self._frame_count,
            "track_hits": self._track_hits,
            "full_searches": self._full_searches,
            "misses": self._total_misses,
            "consecutive_misses": self._consecutive_misses,
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _detect_tracking(
        self,
        frame: MatLike,
        search_mask: Optional[MatLike],
    ) -> Optional[LaserSpot]:
        """在上一帧位置附近搜索，并与目标多边形求交。"""
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
        roi_search_mask = None if search_mask is None else search_mask[y1:y2, x1:x2]
        candidate = self._detect_by_mask(
            roi,
            roi_search_mask,
            expected_position=(lx - x1, ly - y1),
        )
        if candidate is None:
            return None

        return self._candidate_to_spot(candidate, "tracking", offset=(x1, y1))

    def _detect_full(
        self,
        frame: MatLike,
        search_mask: Optional[MatLike],
    ) -> Optional[LaserSpot]:
        """在完整图像或指定目标多边形内搜索。"""
        candidate = self._detect_by_mask(
            frame,
            search_mask,
            expected_position=None,
        )
        if candidate is None:
            return None
        return self._candidate_to_spot(candidate, "full")

    # ------------------------------------------------------------------
    # 检测方法
    # ------------------------------------------------------------------

    def _detect_by_mask(
        self,
        frame: MatLike,
        search_mask: Optional[MatLike] = None,
        expected_position: Optional[Tuple[float, float]] = None,
    ) -> Optional[_LaserCandidate]:
        """直接从 LAB 掩码连通域中选择最可信的激光候选。"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lower, upper = cvt_mvlab2cv(self.threshold)
        mask = cv2.inRange(lab, lower, upper)
        mask = cv2.bitwise_or(mask, self._color_mask(frame))

        if search_mask is not None:
            mask = cv2.bitwise_and(mask, search_mask)

        if self.morph_kernel_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morph_kernel_size, self.morph_kernel_size),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            return None

        best: Optional[_LaserCandidate] = None
        image_h, image_w = gray.shape
        for label in range(1, count):
            x, y, width, height, area = stats[label]
            area = int(area)
            if area < self.min_area or area > self.max_area:
                continue

            aspect_ratio = max(width, height) / max(min(width, height), 1)
            if aspect_ratio > self.max_aspect_ratio:
                continue

            component = labels[y : y + height, x : x + width] == label
            component_gray = gray[y : y + height, x : x + width]
            values = component_gray[component].astype(np.float32)
            if values.size == 0:
                continue

            margin = max(4, self.morph_kernel_size * 2)
            rx1 = max(0, x - margin)
            ry1 = max(0, y - margin)
            rx2 = min(image_w, x + width + margin)
            ry2 = min(image_h, y + height + margin)
            local_labels = labels[ry1:ry2, rx1:rx2]
            ring = local_labels == 0
            if search_mask is not None:
                ring &= search_mask[ry1:ry2, rx1:rx2] > 0
            ring_values = gray[ry1:ry2, rx1:rx2][ring].astype(np.float32)

            if ring_values.size:
                background = float(np.median(ring_values))
                noise = float(np.median(np.abs(ring_values - background))) + 1.0
            else:
                background = 0.0
                noise = 1.0

            peak_value = float(values.max())
            contrast = max(peak_value - background, 0.0)
            snr = contrast / noise

            local_y, local_x = np.nonzero(component)
            weights = np.maximum(values - background, 1.0)
            weight_sum = float(weights.sum())
            center_x = float(x + np.dot(local_x, weights) / weight_sum)
            center_y = float(y + np.dot(local_y, weights) / weight_sum)

            fill_ratio = area / float(max(width * height, 1))
            fill_score = min(fill_ratio / 0.5, 1.0)
            aspect_score = 1.0
            if self.max_aspect_ratio > 1.0:
                aspect_score = 1.0 - min(
                    (aspect_ratio - 1.0) / (self.max_aspect_ratio - 1.0), 1.0
                )
            shape_score = 0.6 * fill_score + 0.4 * aspect_score

            color_score = self._local_color_score(frame[ry1:ry2, rx1:rx2])
            contrast_score = min(contrast / 80.0, 1.0)
            snr_score = min(snr / 12.0, 1.0)
            brightness_score = peak_value / 255.0
            confidence = float(
                np.clip(
                    0.30 * contrast_score
                    + 0.25 * snr_score
                    + 0.20 * color_score
                    + 0.15 * shape_score
                    + 0.10 * brightness_score,
                    0.0,
                    1.0,
                )
            )
            if confidence < self.min_confidence:
                continue

            rank_score = confidence
            if expected_position is not None:
                distance = float(
                    np.hypot(
                        center_x - expected_position[0],
                        center_y - expected_position[1],
                    )
                )
                proximity = max(0.0, 1.0 - distance / self.track_radius)
                rank_score += 0.20 * proximity

            candidate = _LaserCandidate(
                center_x,
                center_y,
                peak_value,
                confidence,
                noise,
                rank_score,
            )
            if best is None or candidate.rank_score > best.rank_score:
                best = candidate

        return best

    def _local_color_score(self, patch: MatLike) -> float:
        values = patch.astype(np.int16)
        blue, green, red = cv2.split(values)
        if self.color_mode == "blue":
            excess = blue - np.maximum(green, red)
        elif self.color_mode == "red":
            excess = red - np.maximum(green, blue)
        else:
            excess = np.maximum(np.maximum(blue, green), red) - np.minimum(
                np.minimum(blue, green), red
            )

        positive = excess[excess > 0].reshape(-1)
        if positive.size == 0:
            return 0.0
        sample_size = min(10, positive.size)
        strongest = np.partition(positive, positive.size - sample_size)[-sample_size:]
        return min(float(strongest.mean()) / 100.0, 1.0)

    def _color_mask(self, frame: MatLike) -> MatLike:
        values = frame.astype(np.int16)
        blue, green, red = cv2.split(values)
        if self.color_mode == "blue":
            dominant = blue
            excess = blue - np.maximum(green, red)
        elif self.color_mode == "red":
            dominant = red
            excess = red - np.maximum(green, blue)
        else:
            dominant = np.maximum(np.maximum(blue, green), red)
            excess = dominant - np.minimum(np.minimum(blue, green), red)
        selected = (dominant >= self.min_color_value) & (
            excess >= self.min_color_excess
        )
        return selected.astype(np.uint8) * 255

    def _build_search_mask(
        self,
        frame_shape: Tuple[int, int],
        search_polygon: Optional[MatLike],
    ) -> Optional[MatLike]:
        if search_polygon is None:
            return None

        points = np.asarray(search_polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3 or not np.all(np.isfinite(points)):
            raise ValueError("search_polygon 必须至少包含 3 个有限坐标点")
        height, width = frame_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
        if self.roi_margin > 0:
            kernel_size = self.roi_margin * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            mask = cv2.erode(mask, kernel)
        return mask

    def _record_miss(self) -> None:
        self._total_misses += 1
        self._consecutive_misses += 1
        if self._consecutive_misses >= self.max_consecutive_misses:
            self._last_position = None
            self._smoothed_position = None

    @staticmethod
    def _candidate_to_spot(
        candidate: _LaserCandidate,
        search_mode: str,
        offset: Tuple[int, int] = (0, 0),
    ) -> LaserSpot:
        return LaserSpot(
            x=candidate.x + offset[0],
            y=candidate.y + offset[1],
            peak_value=candidate.peak_value,
            confidence=candidate.confidence,
            noise_level=candidate.noise_level,
            search_mode=search_mode,
        )

    def _apply_smoothing(self, x: float, y: float) -> Tuple[float, float]:
        if self._smoothed_position is None:
            self._smoothed_position = (x, y)
            return (x, y)
        sx, sy = self._smoothed_position
        if np.hypot(x - sx, y - sy) > self.track_radius:
            self._smoothed_position = (x, y)
            return (x, y)
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
                "area": 0,
                "quad": 0,
                "white_region": 0,
                "aspect_ratio": 0,
            }

        self._frame_count += 1

        use_tracking = self._last_center is not None and (
            self.full_search_interval <= 0
            or self._frame_count % self.full_search_interval != 0
        )

        if use_tracking:
            best_rect = self._track_roi(
                edges,
                gray,
                min_area,
                white_area,
                real_aspect_ratio,
                target_width=target_width,
                tolerance=tolerance,
                epsilon=epsilon,
                reject_status=reject_status,
            )
            if best_rect is None:
                self._misses += 1
                best_rect = detect_rect(
                    edges,
                    gray,
                    min_area,
                    white_area,
                    real_aspect_ratio,
                    target_width=target_width,
                    tolerance=tolerance,
                    epsilon=epsilon,
                    reject_status=reject_status,
                )
            else:
                self._track_hits += 1
        else:
            best_rect = detect_rect(
                edges,
                gray,
                min_area,
                white_area,
                real_aspect_ratio,
                target_width=target_width,
                tolerance=tolerance,
                epsilon=epsilon,
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
            roi_edges,
            roi_gray,
            min_area,
            white_area,
            real_aspect_ratio,
            target_width=target_width,
            tolerance=tolerance,
            epsilon=epsilon,
            reject_status=reject_status,
            num_workers=1,
        )

        if local_rect is None:
            return None

        local_rect = local_rect.astype(np.int32)
        local_rect[:, 0] += x1
        local_rect[:, 1] += y1
        return local_rect
