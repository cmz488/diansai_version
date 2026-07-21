"""自动阈值计算器 — 从目标区域学习 LAB 色彩空间的最佳二值化范围。

策略说明：
  - histogram（默认）：平滑直方图边界法。对采样像素建立 256-bin
    直方图，找主峰后向两侧扩展到最近的低密度边界。
  - mad：中位数绝对偏差法（保留兼容）。
  - percentile：百分位法（保留兼容）。
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

MatLike = np.ndarray


class AutoThresholder:
    """自动阈值计算器 — 直方图边界 / MAD / 百分位策略学习 LAB 范围。"""

    _CHANNEL_NAMES = ["L", "A", "B"]

    def __init__(
        self,
        strategy: str = "histogram",
        mad_factor: float = 3.0,
        min_range: int = 8,
        percentile_low: float = 5.0,
        percentile_high: float = 95.0,
    ) -> None:
        if strategy not in ("histogram", "mad", "percentile"):
            raise ValueError(
                f"strategy 必须是 'histogram'、'mad' 或 'percentile'，实际为 {strategy!r}"
            )
        self.strategy = strategy
        self.mad_factor = mad_factor
        self.min_range = min_range
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self._lower: Optional[np.ndarray] = None
        self._upper: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 学习接口
    # ------------------------------------------------------------------

    def learn_from_rect(
        self, frame: MatLike, x: int, y: int, w: int, h: int
    ) -> "AutoThresholder":
        x = max(0, int(x))
        y = max(0, int(y))
        w = max(1, int(w))
        h = max(1, int(h))
        roi = frame[y : y + h, x : x + w]
        return self._learn_from_pixels(roi)

    def learn_from_center(
        self, frame: MatLike, cx: int, cy: int, half_size: int = 8
    ) -> "AutoThresholder":
        h, w = frame.shape[:2]
        x1 = max(0, int(cx) - half_size)
        y1 = max(0, int(cy) - half_size)
        x2 = min(w, int(cx) + half_size)
        y2 = min(h, int(cy) + half_size)
        roi = frame[y1:y2, x1:x2]
        return self._learn_from_pixels(roi)

    def learn_from_mask(self, frame: MatLike, mask: MatLike) -> "AutoThresholder":
        if mask.ndim != 2 or mask.shape[:2] != frame.shape[:2]:
            raise ValueError("mask 必须是单通道且与 frame 同尺寸")
        pixels = frame[mask > 0]
        if len(pixels) < 3:
            raise RuntimeError("mask 中目标像素不足（<3），无法学习阈值")
        return self._learn_from_pixels(pixels.reshape(-1, 1, 3))

    def learn_from_peaks(
        self,
        frame: MatLike,
        top_n: int = 50,
        half_size: int = 4,
        score_mode: str = "chroma",
    ) -> "AutoThresholder":
        """从候选峰附近学习阈值。

        默认优先选择“具有颜色且较亮”的区域，避免只学习白纸和灯光。
        更可靠的生产用法仍是 ``learn_from_mask`` 或人工确认的 ROI。
        """
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame 必须是 H×W×3 的 BGR 图像")
        if top_n <= 0 or half_size < 1:
            raise ValueError("top_n 必须 > 0，half_size 必须 >= 1")
        if score_mode not in ("chroma", "brightness"):
            raise ValueError("score_mode 必须是 'chroma' 或 'brightness'")

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        if score_mode == "brightness":
            score = l_channel
        else:
            a_chroma = np.abs(lab[:, :, 1].astype(np.float32) - 128.0)
            b_chroma = np.abs(lab[:, :, 2].astype(np.float32) - 128.0)
            score = 0.5 * l_channel + a_chroma + b_chroma

        flat = score.ravel()
        top_n = min(top_n, flat.size)
        indices = np.argpartition(flat, -top_n)[-top_n:]
        indices = indices[np.argsort(flat[indices])[::-1]]
        ys, xs = np.unravel_index(indices, l_channel.shape)

        peaks: List[Tuple[int, int]] = []
        min_dist = half_size * 2
        for px, py in zip(xs, ys):
            if all(
                abs(px - ex) >= min_dist or abs(py - ey) >= min_dist
                for ex, ey in peaks
            ):
                peaks.append((int(px), int(py)))
                if len(peaks) >= 10:
                    break

        h_img, w_img = frame.shape[:2]
        samples = []
        for px, py in peaks:
            x1 = max(0, px - half_size)
            y1 = max(0, py - half_size)
            x2 = min(w_img, px + half_size + 1)
            y2 = min(h_img, py + half_size + 1)
            roi = frame[y1:y2, x1:x2]
            samples.append(roi.reshape(-1, 3))

        if not samples:
            raise RuntimeError("未找到有效峰值")

        all_pixels = np.concatenate(samples, axis=0).reshape(-1, 1, 3)
        return self._learn_from_pixels(all_pixels)

    def _learn_from_pixels(self, pixels_bgr: MatLike) -> "AutoThresholder":
        if pixels_bgr.size == 0:
            raise RuntimeError("采样区域为空，无法学习阈值")
        converted = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
        values = converted.reshape(-1, 3).astype(np.float64)
        if self.strategy == "histogram":
            lower, upper = self._compute_histogram_bounds(values)
        elif self.strategy == "mad":
            lower, upper = self._compute_mad_bounds(values)
        else:
            lower, upper = self._compute_percentile_bounds(values)
        self._lower = np.clip(lower, 0, 255).astype(np.uint8)
        self._upper = np.clip(upper, 0, 255).astype(np.uint8)
        return self

    # ------------------------------------------------------------------
    # 阈值计算
    # ------------------------------------------------------------------

    def _compute_histogram_bounds(
        self, values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """直方图边界法 — 对每通道建立并平滑直方图，
        从主峰向两侧寻找最近的低密度边界。
        histogram 策略使用 cv2.calcHist、GaussianBlur 和 minMaxLoc。"""
        lower = np.zeros(3, dtype=np.float64)
        upper = np.zeros(3, dtype=np.float64)
        for ch in range(3):
            ch_vals = values[:, ch].astype(np.float32)
            lo, hi = self.approx_threshold(
                ch_vals,
                bin_count=256,
                min_range=self.min_range,
            )
            lower[ch] = lo
            upper[ch] = hi
        return lower, upper

    @staticmethod
    def approx_threshold(
        values: np.ndarray,
        bin_count: int = 256,
        min_range: int = 8,
    ) -> Tuple[float, float]:
        """通过平滑直方图检测阈值边界。

        对输入值建立 bin_count-bin 直方图，平滑后从主峰向两侧寻找最近的
        低密度 bin。相比在整个半区寻找绝对梯度最大值，这种方式不会被远处
        的孤立峰或其他颜色模式拉宽阈值范围。

        Args:
            values: 单通道像素值（前景采样），shape (N,)。
            bin_count: 直方图 bin 数，默认 256。
            min_range: 最小上下界间隔（保证阈值范围不会太窄）。

        Returns:
            (lower, upper) 阈值上下界，范围 [0, 255]。

        直方图和平滑使用 OpenCV 原生 API。
        """
        vals = np.asarray(values, dtype=np.float32).ravel()
        n = len(vals)
        if n < 3:
            return 0.0, 255.0

        # ---- 1. 建立直方图 ----
        hist = cv2.calcHist(
            [vals], [0], None, [bin_count], [0.0, 256.0]
        ).ravel().astype(np.float32)

        # ---- 2. 平滑直方图，抑制孤立 bin 噪声 ----
        if bin_count >= 5:
            hist_smooth = cv2.GaussianBlur(
                hist.reshape(-1, 1), (1, 5), 0
            ).ravel()
        else:
            hist_smooth = hist

        # ---- 3. 主峰定位 ----
        _, peak_value, _, max_loc = cv2.minMaxLoc(hist_smooth.reshape(-1, 1))
        peak_bin: int = max_loc[1]

        # ---- 4/5. 从主峰向外找最近的低密度边界 ----
        density_limit = max(float(peak_value) * 0.05, 0.25)
        lower_bin = peak_bin
        while lower_bin > 0 and hist_smooth[lower_bin] > density_limit:
            lower_bin -= 1
        upper_bin = peak_bin
        while upper_bin < bin_count - 1 and hist_smooth[upper_bin] > density_limit:
            upper_bin += 1

        # ---- 6. 保证最小范围 ----
        min_bins = max(1, int(min_range * bin_count / 256))
        if upper_bin - lower_bin < min_bins:
            mid = (lower_bin + upper_bin) / 2.0
            half = min_bins / 2.0
            lower_bin = max(0, int(mid - half))
            upper_bin = min(bin_count - 1, int(mid + half))

        # ---- 7. bin → 实际值 ----
        scale = 255.0 / max(bin_count - 1, 1)
        lo = float(lower_bin) * scale
        hi = float(upper_bin) * scale
        return lo, hi

    def _compute_mad_bounds(
        self, values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(3, dtype=np.float64)
        upper = np.zeros(3, dtype=np.float64)
        for ch in range(3):
            ch_vals = values[:, ch]
            median = float(np.median(ch_vals))
            mad = float(np.median(np.abs(ch_vals - median)))
            half_range = max(self.mad_factor * mad, self.min_range / 2.0)
            lower[ch] = median - half_range
            upper[ch] = median + half_range
        return lower, upper

    def _compute_percentile_bounds(
        self, values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(3, dtype=np.float64)
        upper = np.zeros(3, dtype=np.float64)
        for ch in range(3):
            ch_vals = values[:, ch]
            lo = float(np.percentile(ch_vals, self.percentile_low))
            hi = float(np.percentile(ch_vals, self.percentile_high))
            mid = (lo + hi) / 2.0
            half_range = max((hi - lo) / 2.0, self.min_range / 2.0)
            lower[ch] = mid - half_range
            upper[ch] = mid + half_range
        return lower, upper

    # ------------------------------------------------------------------
    # 应用 & 评估
    # ------------------------------------------------------------------

    @property
    def thresholds(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._lower is None or self._upper is None:
            raise RuntimeError("尚未学习阈值，请先调用 learn_from_* 方法")
        return self._lower.copy(), self._upper.copy()

    @property
    def mvlab_thresholds(self) -> np.ndarray:
        lower, upper = self.thresholds
        return np.array([
            lower[0] / 2.55, upper[0] / 2.55,
            lower[1] - 128,  upper[1] - 128,
            lower[2] - 128,  upper[2] - 128,
        ])

    def apply(self, frame: MatLike) -> MatLike:
        lower, upper = self.thresholds
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        return cv2.inRange(lab, lower, upper)

    def quality(self, frame: MatLike, mask_gt: MatLike) -> Dict[str, float]:
        pred = self.apply(frame)
        tp = float(np.sum((pred > 0) & (mask_gt > 0)))
        fp = float(np.sum((pred > 0) & (mask_gt == 0)))
        fn = float(np.sum((pred == 0) & (mask_gt > 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}

    def describe(self) -> str:
        if self._lower is None or self._upper is None:
            return "AutoThresholder(未学习)"
        parts = []
        for i, name in enumerate(self._CHANNEL_NAMES):
            parts.append(f"{name}=[{self._lower[i]}, {self._upper[i]}]")
        return f"AutoThresholder({self.strategy}): " + ", ".join(parts)
