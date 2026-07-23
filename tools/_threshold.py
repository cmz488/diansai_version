"""自动二值化模块 — 统一的图像二值化深度接口。

提供两种策略，通过一个 apply() 方法隐藏全部实现细节：

  - adaptive（默认）：灰度自适应阈值。
    根据画面平均亮度自动计算 cv2.adaptiveThreshold 所需的 block_size 和 C，
    随后进行高斯模糊 → 自适应阈值 → 形态学闭+开运算去噪填洞。
    适合光照不均场景（激光点、光斑检测）。

  - range：LAB 色彩空间范围阈值。
    先从 ROI 采样学习目标颜色的 LAB 范围（直方图边界法），
    再用 cv2.inRange 精确分割。
    适合颜色区分度高的目标（特定色块、标记物）。

Usage::

    # 自适应模式（零配置）
    b = Binarizer()
    mask = b.apply(frame)

    # 范围学习模式
    b = Binarizer(strategy="range")
    b.learn(frame, roi_x=100, roi_y=50, roi_w=40, roi_h=40)
    mask = b.apply(frame)
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

MatLike = np.ndarray

# =========================================================================
# 亮度 → adaptiveThreshold 参数映射
# =========================================================================


def auto_paramC(brightness: float) -> Tuple[float, int]:
    """根据灰度图像的平均亮度，动态计算 adaptiveThreshold 的 C 和 block_size。

    规律：
      - 画面越亮 → block_size 和 C 越大（邻域更广、阈值更低 / 更宽松）。
      - 画面越暗 → block_size 和 C 越小（邻域更窄、阈值更高 / 更严格）。

    原理：亮场景对比度高、噪声可见，需要更大的邻域来平滑、更大的 C
    来降低阈值捕获目标；暗场景对比度低，需要更精细的局部处理。

    Args:
        brightness: 画面平均亮度 (0~255)，通常取自 GaussianBlur 后的 mean。

    Returns:
        (C, block_size) — C 范围 2~10，block_size 范围 3~21（奇数）。
    """
    brightness = float(np.clip(brightness, 0, 255))
    ratio = brightness / 255.0

    # block_size: 3~21，强制奇数
    block_size = int(3.0 + ratio * 18.0)
    if block_size % 2 == 0:
        block_size += 1

    # C: 2~10
    C = int(2.0 + ratio * 8.0)

    return C, block_size


# =========================================================================
# 共享工具：直方图边界法
# =========================================================================


def _adaptive_bin_count(n_samples: int) -> int:
    """样本量 → 最优 bin 数。样本少时降低 bin 数以抑制直方图噪声。"""
    if n_samples < 30:
        return max(8, n_samples // 2)
    if n_samples < 200:
        return max(16, n_samples // 3)
    if n_samples < 1000:
        return max(32, n_samples // 5)
    return 256


def _adaptive_smooth_kernel(bin_count: int) -> int:
    """bin 数 → 高斯平滑核半宽（奇数）。bin 多时需要更宽的平滑核。"""
    k = max(1, bin_count // 50)
    return k if k % 2 == 1 else k + 1


def _reject_outliers(values: np.ndarray, iqr_mult: float = 2.5) -> np.ndarray:
    """IQR 离群值剔除：去掉 ±2.5×IQR 外的极端像素。"""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr < 1e-6:
        return values  # 纯色，无需剔除
    lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
    return values[(values >= lo) & (values <= hi)]


def _find_significant_peaks(
    hist_smooth: np.ndarray, noise_floor: float = 0.0
) -> list[int]:
    """检测直方图中所有显著峰（局部最大值，高于噪声地板）。"""
    n = len(hist_smooth)
    if n < 3:
        return []
    peaks: list[int] = []
    for i in range(1, n - 1):
        if hist_smooth[i] > hist_smooth[i - 1] and hist_smooth[i] > hist_smooth[i + 1]:
            if hist_smooth[i] > noise_floor:
                peaks.append(i)
    return peaks


def approx_threshold(
    values: np.ndarray,
    bin_count: int = 256,
    min_range: int = 8,
) -> Tuple[float, float]:
    """通过平滑直方图检测单通道数据的阈值边界。

    对输入值建立 bin_count-bin 直方图，平滑后从主峰向两侧寻找最近的
    低密度 bin。相比寻找绝对梯度最大值，不会被远处的孤立峰拉宽范围。

    Args:
        values: 单通道数据，shape (N,)。
        bin_count: 直方图 bin 数，默认 256。
        min_range: 最小上下界间隔（保证阈值范围不会太窄）。

    Returns:
        (lower, upper) 阈值上下界，范围 [0, 255]。

    直方图和平滑使用 OpenCV 原生 API（calcHist / GaussianBlur / minMaxLoc）。
    """
    vals = np.asarray(values, dtype=np.float32).ravel()
    n = len(vals)
    if n < 3:
        return 0.0, 255.0

    # 1. 建立直方图
    hist = (
        cv2.calcHist([vals], [0], None, [bin_count], [0.0, 256.0])
        .ravel()
        .astype(np.float32)
    )

    # 2. 平滑直方图
    if bin_count >= 5:
        hist_smooth = cv2.GaussianBlur(hist.reshape(-1, 1), (1, 5), 0).ravel()
    else:
        hist_smooth = hist

    # 3. 主峰定位
    _, peak_value, _, max_loc = cv2.minMaxLoc(hist_smooth.reshape(-1, 1))
    peak_bin: int = max_loc[1]

    # 4/5. 从主峰向外找最近的低密度边界
    density_limit = max(float(peak_value) * 0.05, 0.25)
    lower_bin = peak_bin
    while lower_bin > 0 and hist_smooth[lower_bin] > density_limit:
        lower_bin -= 1
    upper_bin = peak_bin
    while upper_bin < bin_count - 1 and hist_smooth[upper_bin] > density_limit:
        upper_bin += 1

    # 6. 保证最小范围
    min_bins = max(1, int(min_range * bin_count / 256))
    if upper_bin - lower_bin < min_bins:
        mid = (lower_bin + upper_bin) / 2.0
        half = min_bins / 2.0
        lower_bin = max(0, int(mid - half))
        upper_bin = min(bin_count - 1, int(mid + half))

    # 7. bin → 实际值
    scale = 255.0 / max(bin_count - 1, 1)
    lo = float(lower_bin) * scale
    hi = float(upper_bin) * scale
    return lo, hi


def approx_threshold_v2(
    values: np.ndarray,
    min_range: int = 8,
    outlier_iqr: float = 2.5,
    valley_ratio: float = 0.10,
) -> Tuple[float, float]:
    """改进版直方图边界检测。

    优化点（相对于 approx_threshold）：
    1. 自适应 bin 数 — 根据样本量动态选择，避免小样本稀疏直方图
    2. 自适应平滑核 — bin 数 → 核宽度自动匹配
    3. 离群值剔除 — IQR 方法去掉极端像素，防止拉宽范围
    4. 多峰合并 — 检测所有显著峰，合并它们的覆盖范围
    5. 谷底边界 — 在峰群两侧找最近谷底，而非固定密度阈值

    Args:
        values: 单通道数据，shape (N,)。
        min_range: 最小上下界间隔。
        outlier_iqr: 离群值 IQR 倍数（默认 2.5）。
        valley_ratio: 峰高 × valley_ratio 以下视为谷底。

    Returns:
        (lower, upper) 阈值上下界，范围 [0, 255]。
    """
    vals = np.asarray(values, dtype=np.float32).ravel()
    n_orig = len(vals)
    if n_orig < 5:
        return 0.0, 255.0

    # --- 1. 离群值剔除 ---
    cleaned = _reject_outliers(vals, iqr_mult=outlier_iqr)
    n = len(cleaned)
    if n < 5:
        return 0.0, 255.0

    # --- 2. 自适应 bin 数 + 平滑核 ---
    bin_count = _adaptive_bin_count(n)
    smooth_k = _adaptive_smooth_kernel(bin_count)

    hist = (
        cv2.calcHist([cleaned], [0], None, [bin_count], [0.0, 256.0])
        .ravel()
        .astype(np.float32)
    )

    if bin_count >= 5 and smooth_k >= 3:
        hist_smooth = cv2.GaussianBlur(hist.reshape(-1, 1), (1, smooth_k), 0).ravel()
    else:
        hist_smooth = hist

    # --- 3. 多峰检测 ---
    noise_floor = float(np.max(hist_smooth)) * 0.02
    peaks = _find_significant_peaks(hist_smooth, noise_floor)

    if not peaks:
        # 退化：取包含 95% 样本的范围
        lo_val = float(np.percentile(cleaned, 2.5))
        hi_val = float(np.percentile(cleaned, 97.5))
        return max(0, lo_val), min(255, hi_val)

    # --- 4. 峰群合并：从最高峰向两边扩展，找谷底 ---
    peak_vals = hist_smooth[peaks]
    main_idx = peaks[int(np.argmax(peak_vals))]
    max_val = float(hist_smooth[main_idx])
    valley_threshold = max_val * valley_ratio

    # 向左找谷底（局部最小值，或跌破 valley_threshold）
    left_bin = main_idx
    while left_bin > 0:
        if hist_smooth[left_bin] < valley_threshold:
            break
        # 如果下降后又上升 → 到了另一个峰的谷底
        if (
            left_bin < main_idx - 2
            and hist_smooth[left_bin] > hist_smooth[left_bin - 1]
            and hist_smooth[left_bin - 1] < hist_smooth[left_bin - 2]
        ):
            break
        left_bin -= 1

    # 向右找谷底
    right_bin = main_idx
    while right_bin < bin_count - 2:
        if hist_smooth[right_bin] < valley_threshold:
            break
        if (
            right_bin > main_idx + 2
            and hist_smooth[right_bin] > hist_smooth[right_bin + 1]
            and hist_smooth[right_bin + 1] < hist_smooth[right_bin + 2]
        ):
            break
        right_bin += 1

    # --- 5. 合并附近的其他显著峰 ---
    peak_threshold = max_val * 0.20
    for p in peaks:
        if hist_smooth[p] >= peak_threshold:
            left_bin = min(left_bin, p)
            right_bin = max(right_bin, p)

    # --- 6. 最小范围 ---
    min_bins = max(1, int(min_range * bin_count / 256))
    if right_bin - left_bin < min_bins:
        mid = (left_bin + right_bin) / 2.0
        half = min_bins / 2.0
        left_bin = max(0, int(mid - half))
        right_bin = min(bin_count - 1, int(mid + half))

    # --- 7. bin → 值 ---
    scale = 255.0 / max(bin_count - 1, 1)
    lo = float(left_bin) * scale
    hi = float(right_bin) * scale
    return lo, hi


def _compute_histogram_bounds(
    values: np.ndarray, min_range: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """对 LAB 三通道分别计算直方图边界。"""
    lower = np.zeros(3, dtype=np.float64)
    upper = np.zeros(3, dtype=np.float64)
    for ch in range(3):
        ch_vals = values[:, ch].astype(np.float32)
        lo, hi = approx_threshold(ch_vals, bin_count=256, min_range=min_range)
        lower[ch] = lo
        upper[ch] = hi
    return lower, upper


def _compute_histogram_bounds_v2(
    values: np.ndarray,
    min_range_l: int = 10,
    min_range_ab: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """改进版三通道直方图边界 + 质量指标。

    L 通道和 A/B 通道使用不同的 min_range（L 分布更宽）。
    返回 (lower, upper, quality_dict)。
    """
    lower = np.zeros(3, dtype=np.float64)
    upper = np.zeros(3, dtype=np.float64)
    quality: Dict[str, float] = {}
    min_ranges = [min_range_l, min_range_ab, min_range_ab]

    for ch in range(3):
        ch_vals = values[:, ch].astype(np.float32)
        lo, hi = approx_threshold_v2(ch_vals, min_range=min_ranges[ch])
        lower[ch] = lo
        upper[ch] = hi

    # 质量指标
    span = upper - lower
    quality["span_L"] = float(span[0])
    quality["span_AB"] = float(max(span[1], span[2]))
    quality["std_L"] = float(np.std(values[:, 0]))
    # 标准差太小 → 可能是纯色背景
    quality["is_informative"] = float(
        1.0 if quality["std_L"] > 3.0 or quality["span_AB"] > 5.0 else 0.0
    )

    return lower, upper, quality


def _compute_mad_bounds(
    values: np.ndarray, mad_factor: float = 3.0, min_range: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """中位数绝对偏差法 — 按 MAD * factor 扩展。"""
    lower = np.zeros(3, dtype=np.float64)
    upper = np.zeros(3, dtype=np.float64)
    for ch in range(3):
        ch_vals = values[:, ch]
        median = float(np.median(ch_vals))
        mad = float(np.median(np.abs(ch_vals - median)))
        half_range = max(mad_factor * mad, min_range / 2.0)
        lower[ch] = median - half_range
        upper[ch] = median + half_range
    return lower, upper


def _compute_percentile_bounds(
    values: np.ndarray,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
    min_range: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """百分位法 — 按低/高百分位对称扩展。"""
    lower = np.zeros(3, dtype=np.float64)
    upper = np.zeros(3, dtype=np.float64)
    for ch in range(3):
        ch_vals = values[:, ch]
        lo = float(np.percentile(ch_vals, percentile_low))
        hi = float(np.percentile(ch_vals, percentile_high))
        mid = (lo + hi) / 2.0
        half_range = max((hi - lo) / 2.0, min_range / 2.0)
        lower[ch] = mid - half_range
        upper[ch] = mid + half_range
    return lower, upper


# =========================================================================
# 新类：Binarizer — 深度二值化模块
# =========================================================================


class Binarizer:
    """自动二值化器 — 统一的图像二值化深度模块。

    ┌──────────────────────────────────────┐
    │  apply(frame) → mask (uint8 0/255)   │  ← 唯一对外接口
    │  learn(frame, roi=...) → self        │  ← 可选学习（range 策略）
    ├──────────────────────────────────────┤
    │  策略路由 + 参数自动调优               │
    │  • adaptive: 灰度自适应               │
    │    auto_paramC(亮度) → GaussianBlur   │
    │    → adaptiveThreshold → 形态学后处理  │
    │  • range: LAB 色彩范围                 │
    │    learn → 直方图边界法 → inRange      │
    └──────────────────────────────────────┘

    自适应模式（默认，零配置）::

        b = Binarizer()
        mask = b.apply(frame)

    范围学习模式::

        b = Binarizer(strategy="range")
        b.learn(frame, roi_x=100, roi_y=50, roi_w=40, roi_h=40)
        mask = b.apply(frame)
    """

    _CHANNEL_NAMES = ["L", "A", "B"]
    _VALID_STRATEGIES = ("adaptive", "range")

    def __init__(
        self,
        strategy: str = "adaptive",
        *,
        morph_kernel: Tuple[int, int] = (3, 3),
        min_range: int = 8,
        gauss_kernel: Tuple[int, int] = (3, 3),
    ) -> None:
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"strategy 必须是 {self._VALID_STRATEGIES}，实际为 {strategy!r}"
            )
        self.strategy = strategy
        self.morph_kernel = morph_kernel
        self.min_range = min_range
        self.gauss_kernel = gauss_kernel
        self._lower: Optional[np.ndarray] = None
        self._upper: Optional[np.ndarray] = None
        self._last_C: Optional[float] = None
        self._last_block_size: Optional[int] = None
        self._quality: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def apply(self, frame: MatLike) -> MatLike:
        """将 BGR 帧二值化，返回 uint8 掩码。

        Args:
            frame: BGR 三通道图像 (H×W×3)。

        Returns:
            二值掩码 (H×W) uint8，前景=255，背景=0。
        """
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame 必须是 H×W×3 的 BGR 图像")

        if self.strategy == "adaptive":
            return self._apply_adaptive(frame)
        else:
            return self._apply_range(frame)

    def learn(
        self,
        frame: MatLike,
        *,
        roi_x: int,
        roi_y: int,
        roi_w: int,
        roi_h: int,
    ) -> "Binarizer":
        """从帧中指定 ROI 区域采样，用直方图边界法学习 LAB 色彩范围。

        Args:
            frame: BGR 三通道图像 (H×W×3)。
            roi_x, roi_y: ROI 左上角坐标。
            roi_w, roi_h: ROI 宽高。

        Returns:
            self，支持链式调用。
        """
        x = max(0, int(roi_x))
        y = max(0, int(roi_y))
        w = max(1, int(roi_w))
        h = max(1, int(roi_h))
        roi = frame[y : y + h, x : x + w]
        self._learn_from_pixels(roi)
        return self

    def learn_from_rect_pts(
        self, frame: MatLike, pts: MatLike, *, margin: int = 5
    ) -> "Binarizer":
        """从矩形角点自动提取 ROI 学习 LAB 色彩范围。

        与 RectTracker 等追踪器配合使用：追踪器检测到矩形后，
        将四个角点传入即可自动学习目标颜色范围。

        Args:
            frame: BGR 三通道图像 (H×W×3)。
            pts: 矩形四个角点，shape (4, 2)，顺序不限。
            margin: ROI 向内收缩边距（避免包含背景边缘）。

        Returns:
            self，支持链式调用。
        """
        pts_i = np.asarray(pts, dtype=np.int32).reshape(4, 2)
        x, y, w, h = cv2.boundingRect(pts_i)
        m = max(0, int(margin))
        return self.learn(
            frame,
            roi_x=x + m,
            roi_y=y + m,
            roi_w=max(1, w - 2 * m),
            roi_h=max(1, h - 2 * m),
        )

    @property
    def is_learned(self) -> bool:
        """是否已完成学习（range 策略可用）。"""
        return self._lower is not None and self._upper is not None

    @property
    def params(self) -> Dict[str, Any]:
        """当前运行时参数，调试用。"""
        info: Dict[str, Any] = {"strategy": self.strategy}
        if self.strategy == "adaptive":
            info["C"] = self._last_C
            info["block_size"] = self._last_block_size
            info["morph_kernel"] = self.morph_kernel
            info["gauss_kernel"] = self.gauss_kernel
        else:
            info["lower"] = self._lower if self._lower is not None else None
            info["upper"] = self._upper if self._upper is not None else None
            info["quality"] = self._quality
        return info

    @property
    def quality(self) -> Dict[str, float]:
        """学习质量指标（仅 range 策略）。

        span_L:     L 通道阈值跨度（>10 较好）
        span_AB:    AB 通道最大跨度（>5 表示有色度信息）
        std_L:      采样像素 L 通道标准差（>3 表示非纯色）
        is_informative: 采样是否包含有效信息（1.0=是, 0.0=可能选中了纯色背景）
        """
        return dict(self._quality)

    def describe(self) -> str:
        """人类可读的参数摘要。"""
        if self.strategy == "adaptive":
            return (
                f"Binarizer(adaptive): C={self._last_C}, "
                f"block_size={self._last_block_size}, "
                f"morph={self.morph_kernel}, gauss={self.gauss_kernel}"
            )
        if self._lower is None or self._upper is None:
            return "Binarizer(range): 未学习"
        parts = []
        for i, name in enumerate(self._CHANNEL_NAMES):
            parts.append(f"{name}=[{self._lower[i]}, {self._upper[i]}]")
        return "Binarizer(range): " + ", ".join(parts)

    # ------------------------------------------------------------------
    # 内部：自适应阈值管线
    # ------------------------------------------------------------------

    def _apply_adaptive(self, frame: MatLike) -> MatLike:
        """灰度自适应阈值全管线。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.gauss_kernel, 0)
        # 计算灰度图平均亮度
        C, block_size = auto_paramC(float(np.mean(gray)))
        self._last_C = C
        self._last_block_size = block_size

        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morph_kernel)
        threshold = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # 高斯滤波后一般没有小的噪点
        # threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        return threshold

    # ------------------------------------------------------------------
    # 内部：色彩范围阈值
    # ------------------------------------------------------------------

    def _apply_range(self, frame: MatLike) -> MatLike:
        if self._lower is None or self._upper is None:
            raise RuntimeError("尚未学习阈值，请先调用 learn()")
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        return cv2.inRange(lab, self._lower, self._upper)

    def _learn_from_pixels(self, pixels_bgr: MatLike) -> None:
        """从 BGR 像素学习 LAB 范围（改进版直方图边界法 + 离群值剔除 + 多峰合并）。"""
        if pixels_bgr.size == 0:
            raise RuntimeError("采样区域为空，无法学习阈值")
        converted = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
        values = converted.reshape(-1, 3).astype(np.float64)
        lower, upper, quality = _compute_histogram_bounds_v2(values)
        self._lower = np.clip(lower, 0, 255).astype(np.uint8)
        self._upper = np.clip(upper, 0, 255).astype(np.uint8)
        self._quality = quality


# =========================================================================
# 旧类：AutoThresholder — 向后兼容
# =========================================================================


class AutoThresholder(Binarizer):
    """@deprecated: 请迁移到 Binarizer。

    保留以兼容现有代码（detect_laser_binary、test_laser_detect 等）。

    旧参数映射:
      - strategy:   histogram → Binarizer(strategy="range") + 直方图学习
                    mad       → Binarizer(strategy="range") + MAD 学习
                    percentile → Binarizer(strategy="range") + 百分位学习
    """

    _LEARN_STRATEGIES = ("histogram", "mad", "percentile")

    def __init__(
        self,
        strategy: str = "histogram",
        mad_factor: float = 3.0,
        min_range: int = 8,
        percentile_low: float = 5.0,
        percentile_high: float = 95.0,
    ) -> None:
        if strategy not in self._LEARN_STRATEGIES:
            raise ValueError(
                f"strategy 必须是 {self._LEARN_STRATEGIES}，实际为 {strategy!r}"
            )
        # 底层统一用 range 策略
        super().__init__(strategy="range", min_range=min_range)
        self.learn_strategy = strategy
        self.mad_factor = mad_factor
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high

    # ------------------------------------------------------------------
    # 旧学习接口（委托到 learn / _learn_from_pixels）
    # ------------------------------------------------------------------

    def learn_from_rect(
        self, frame: MatLike, x: int, y: int, w: int, h: int
    ) -> "AutoThresholder":
        x = max(0, int(x))
        y = max(0, int(y))
        w = max(1, int(w))
        h = max(1, int(h))
        roi = frame[y : y + h, x : x + w]
        return self._learn_from_pixels(roi)  # type: ignore[return-value]

    def learn_from_center(
        self, frame: MatLike, cx: int, cy: int, half_size: int = 8
    ) -> "AutoThresholder":
        h, w = frame.shape[:2]
        x1 = max(0, int(cx) - half_size)
        y1 = max(0, int(cy) - half_size)
        x2 = min(w, int(cx) + half_size)
        y2 = min(h, int(cy) + half_size)
        roi = frame[y1:y2, x1:x2]
        return self._learn_from_pixels(roi)  # type: ignore[return-value]

    def learn_from_mask(self, frame: MatLike, mask: MatLike) -> "AutoThresholder":
        if mask.ndim != 2 or mask.shape[:2] != frame.shape[:2]:
            raise ValueError("mask 必须是单通道且与 frame 同尺寸")
        pixels = frame[mask > 0]
        if len(pixels) < 3:
            raise RuntimeError("mask 中目标像素不足（<3），无法学习阈值")
        return self._learn_from_pixels(pixels.reshape(-1, 1, 3))  # type: ignore[return-value]

    def learn_from_peaks(
        self,
        frame: MatLike,
        top_n: int = 50,
        half_size: int = 4,
        score_mode: str = "chroma",
    ) -> "AutoThresholder":
        """从候选峰附近学习阈值。

        默认优先选择"具有颜色且较亮"的区域，避免只学习白纸和灯光。
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
                abs(px - ex) >= min_dist or abs(py - ey) >= min_dist for ex, ey in peaks
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
        return self._learn_from_pixels(all_pixels)  # type: ignore[return-value]

    def _learn_from_pixels(self, pixels_bgr: MatLike) -> "AutoThresholder":
        """覆盖 Binarizer 的学习逻辑，支持 histogram/mad/percentile 三种策略。"""
        if pixels_bgr.size == 0:
            raise RuntimeError("采样区域为空，无法学习阈值")
        converted = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
        values = converted.reshape(-1, 3).astype(np.float64)

        if self.learn_strategy == "histogram":
            lower, upper = _compute_histogram_bounds(values, self.min_range)
        elif self.learn_strategy == "mad":
            lower, upper = _compute_mad_bounds(values, self.mad_factor, self.min_range)
        else:
            lower, upper = _compute_percentile_bounds(
                values, self.percentile_low, self.percentile_high, self.min_range
            )

        self._lower = np.clip(lower, 0, 255).astype(np.uint8)
        self._upper = np.clip(upper, 0, 255).astype(np.uint8)
        return self

    # ------------------------------------------------------------------
    # 旧属性 & 方法
    # ------------------------------------------------------------------

    @property
    def thresholds(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (lower, upper) OpenCV LAB uint8 阈值。"""
        if self._lower is None or self._upper is None:
            raise RuntimeError("尚未学习阈值，请先调用 learn_from_* 方法")
        return self._lower.copy(), self._upper.copy()

    @property
    def mvlab_thresholds(self) -> np.ndarray:
        """返回 Machine Vision LAB 格式 [Lmin, Lmax, Amin, Amax, Bmin, Bmax]。"""
        lower, upper = self.thresholds
        return np.array(
            [
                float(lower[0]) / 2.55,
                float(upper[0]) / 2.55,
                int(lower[1]) - 128,
                int(upper[1]) - 128,
                int(lower[2]) - 128,
                int(upper[2]) - 128,
            ]
        )

    def quality(self, frame: MatLike, mask_gt: MatLike) -> Dict[str, float]:
        """评估二值化质量：precision, recall, f1, iou。"""
        pred = self.apply(frame)
        tp = float(np.sum((pred > 0) & (mask_gt > 0)))
        fp = float(np.sum((pred > 0) & (mask_gt == 0)))
        fn = float(np.sum((pred == 0) & (mask_gt > 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}

    # approx_threshold 现在是模块级函数，此处作为静态方法暴露以兼容旧引用
    approx_threshold = staticmethod(approx_threshold)

    def describe(self) -> str:
        if self._lower is None or self._upper is None:
            return "AutoThresholder(未学习)"
        parts = []
        for i, name in enumerate(self._CHANNEL_NAMES):
            parts.append(f"{name}=[{self._lower[i]}, {self._upper[i]}]")
        return f"AutoThresholder({self.learn_strategy}): " + ", ".join(parts)


# =========================================================================
# 便捷函数：灰度自适应二值化（保留兼容）
# =========================================================================


def auto_threshold(gray: MatLike, kernel_size: Tuple[int, int]) -> MatLike:
    """通过灰度图直接自适应二值化，返回二值化图像。

    这是 auto_paramC + adaptiveThreshold + 形态学处理的便捷封装。
    新代码推荐直接使用::

        Binarizer(morph_kernel=kernel_size, gauss_kernel=kernel_size).apply(frame)

    Args:
        gray: 单通道灰度图 (H×W)。
        kernel_size: 高斯核和形态学核尺寸。

    Returns:
        二值掩码 (H×W) uint8。
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, kernel_size, 0)
    C, block_size = auto_paramC(float(np.mean(blurred)))

    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return cleaned
