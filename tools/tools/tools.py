"""
图像处理工具模块 — 提供矩形检测、激光遮罩检测、透视校正与预处理等功能。

本模块是 TSP（任务调度平台）视觉识别管线的核心工具集，主要服务于
车牌 / 矩形标牌的检测与校正场景。各个函数按照处理链路组织：

    preprocess() → detect_rect() → perspective_correct_and_validate()
                  → detect_laser_mask()

"""

from __future__ import annotations

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from cv2 import MatLike
except ImportError:
    # Ubuntu 22.04 / OpenCV 4.5.4 does not export cv2.MatLike.
    MatLike = np.ndarray

# ============================================================================
# 常量定义
# ============================================================================

# 矩形检测的默认 LAB 色彩空间阈值（Machine Vision LAB 格式）
# 格式: [L_min, L_max, A_min, A_max, B_min, B_max]
#   - L 通道: 亮度 (0–100)，该默认值对应较为明亮的白色区域
#   - A 通道: 绿↔品红 (-128 ~ +127)
#   - B 通道: 蓝↔黄    (-128 ~ +127)
# 这些值来自机器视觉软件（如 OpenMV）常用的 LAB 表示法，
# L 为 0–100 百分制，A/B 为 -128~127 有符号整型，
# 与 OpenCV 的 0–255 表示法不同，需要通过 cvt_mvlab2cv() 转换。
RECT_MV_LABVALUE = [7, 32, -13, 9, -13, 21]


# ============================================================================
# 工具类
# ============================================================================


class FpsShow:
    """实时 FPS 显示工具。

    在视频帧上叠加当前帧率的文字标注，用于性能监控与调试。
    每次调用 show() 都会基于上一次调用时间戳计算瞬时 FPS。

    使用示例:
        fps_show = FpsShow()
        while True:
            frame = camera.read()
            frame = fps_show.show(frame)
            cv2.imshow("preview", frame)
    """

    def __init__(self, smoothing: float = 0.9) -> None:
        """初始化 FPS 计时器，记录当前时间为基准时间戳。"""
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing 必须位于 [0, 1)")
        self.last_ = time.perf_counter()
        self.smoothing = smoothing
        self.fps = 0.0

    def show(self, frame: MatLike) -> MatLike:
        """在给定帧上绘制 FPS 文字并返回。

        计算逻辑:
            fps = 1 / Δt，其中 Δt 为距离上一次调用的时间间隔（秒）。

        文字位置:
            位于图像右下区域，大约是 (80% 宽度, 20% 高度) 处，
            使用绿色粗体字（PLAIN 字体 + 字号 3）。

        参数:
            frame: 输入图像帧（BGR 格式），会直接在其上绘制（原地修改）

        返回:
            标注了 FPS 文字的同一帧对象（原地修改的引用）
        """
        current = time.perf_counter()
        elapsed = max(current - self.last_, 1e-9)
        instant_fps = 1.0 / elapsed
        self.last_ = current
        self.fps = (
            instant_fps
            if self.fps == 0.0
            else self.smoothing * self.fps + (1.0 - self.smoothing) * instant_fps
        )
        h = frame.shape[0]  # 图像高度（像素行数）
        w = frame.shape[1]  # 图像宽度（像素列数）

        frame = cv2.putText(
            frame,
            "fps:{:.2f}".format(self.fps),  # 保留两位小数的帧率
            (int(w * 0.8), int(h * 0.2)),  # 标注位置：右下区域
            cv2.FONT_HERSHEY_PLAIN,  # 无衬线等宽字体
            3,  # 字体缩放因子
            (0, 255, 0),  # 绿色 (B=0, G=255, R=0)
        )

        return frame


# ============================================================================
# 色彩空间转换
# ============================================================================


def cvt_mvlab2cv(param: Sequence[float] = RECT_MV_LABVALUE):
    """将机器视觉 LAB 格式的阈值转换为 OpenCV LAB 格式。

    两种 LAB 表示法的差异:
        - MV LAB:  L ∈ [0,   100],  A/B ∈ [-128, 127]（有符号）
        - CV LAB:  L ∈ [0,   255],  A/B ∈ [   0, 255]（无符号）

    转换公式:
        L_cv = L_mv × 2.55          （0→0, 100→255）
        A_cv = A_mv + 128            （-128→0, 0→128, 127→255）
        B_cv = B_mv + 128            （同上）

    参数:
        param: 长度为 6 的数组 [lmin, lmax, amin, amax, bmin, bmax]，
               按机器视觉 LAB 表示法给出

    返回:
        (lower_bound, upper_bound): 两个 (3,) 形状的 uint8 numpy 数组，
        可直接用于 cv2.inRange()
    """
    lmin, lmax, amin, amax, bmin, bmax = param
    lower_bound = np.array(
        [int(lmin * 2.55), int(amin + 128), int(bmin + 128)], dtype=np.uint8
    )
    upper_bound = np.array(
        [int(lmax * 2.55), int(amax + 128), int(bmax + 128)], dtype=np.uint8
    )
    return lower_bound, upper_bound


# ============================================================================
# 几何工具
# ============================================================================


def order_points(pts: np.ndarray) -> np.ndarray:
    """四点排序：左上 → 右上 → 右下 → 左下。

    此函数是透视校正的前置步骤——cv2.getPerspectiveTransform 要求
    源点和目标点按一致顺序排列（顺时针或逆时针），本函数统一为顺时针。

    排序原理:
        - 左上点:  x + y 之和最小 （离原点最近）
        - 右下点:  x + y 之和最大 （离原点最远）
        - 右上点:  x - y 之差最大 （偏右偏上）
        - 左下点:  x - y 之差最小 （偏左偏下）

    参数:
        pts: 形状 (4, 2) 的 numpy 数组，四个角点的像素坐标，允许任意顺序

    返回:
        形状 (4, 2) 的 int32 numpy 数组，按 [左上, 右上, 右下, 左下] 排列
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # 利用 x+y 的和区分对角点
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上 — 和最小
    rect[2] = pts[np.argmax(s)]  # 右下 — 和最大

    # 利用 x-y 的差区分另外两个角点
    diff = np.diff(pts, axis=1)  # axis=1 即对每行计算 x - y
    rect[1] = pts[np.argmax(diff)]  # 右上 — 差最大
    rect[3] = pts[np.argmin(diff)]  # 左下 — 差最小

    return rect.astype(np.int32)


# ============================================================================
# 透视校正与验证
# ============================================================================


def perspective_correct_and_validate(
    image: np.ndarray,
    pts: np.ndarray,
    real_aspect_ratio: float,
    target_width: Optional[int] = None,
    tolerance: float = 0.1,
) -> "tuple[np.ndarray, bool, float]":
    """对四点围成的四边形区域做透视校正，并验证其像素宽高比是否接近真实值。

    典型应用场景:
        从图像中检测到一个倾斜的矩形区域（如车牌、标牌），将其
        校正为正向矩形，同时检查该区域的形状是否合理。

    与常见实现的关键区别:
        - 验证的是 **源四边形** 在图像中的像素宽高比，而非校正后的目标图
          （目标图尺寸由 real_aspect_ratio 决定，验证它没有意义——恒为 True）
        - target_width 默认自动取源四边形的平均宽度，保留原始分辨率

    处理流程:
        1. 对输入四点排序（左上→右上→右下→左下）
        2. 计算源四边形的平均宽度和高度（取对边平均以抗透视畸变）
        3. 将源宽高比与真实物理宽高比进行比较
        4. 若通过验证（或无需验证），执行透视变换输出校正图像

    参数:
        image             : 输入图像（BGR 或灰度均可）
        pts               : 四个角点，形状 (4, 2)，允许任意顺序
        real_aspect_ratio : 真实物理宽高比 (width / height)
                            例如中国蓝牌 ≈ 440/140 ≈ 3.14
        target_width      : 输出图像宽度（像素）。None 则自动取源四边形平均宽度
        tolerance         : 允许的相对误差，默认 0.1 表示 ±10%

    返回:
        warped       : 透视校正后的图像（通道数与输入一致，dtype 与输入一致）
        is_valid     : 源四边形像素宽高比是否在容忍范围内
        actual_ratio : 源四边形实际计算出的像素宽高比（h_src < 1 时为 inf）
    """
    # ---- 输入校验 ----
    if pts.shape != (4, 2):
        raise ValueError(f"pts 形状必须为 (4, 2)，实际为 {pts.shape}")
    if real_aspect_ratio <= 0:
        raise ValueError(f"real_aspect_ratio 必须 > 0，实际为 {real_aspect_ratio}")
    if target_width is not None and target_width <= 0:
        raise ValueError(f"target_width 必须 > 0，实际为 {target_width}")

    # ---- 1. 排序并转为 float32 ----
    # 透视变换矩阵计算需要 float32 精度，int 会导致精度损失
    pts_src = order_points(pts).astype(np.float32)

    # ---- 2. 计算源四边形在图像中的像素宽度和高度 ----
    # 用对边平均来抗透视畸变带来的边长差异：
    # - 上边 (pts[0]→pts[1]) 和下边 (pts[3]→pts[2]) 的平均作为宽度
    # - 左边 (pts[0]→pts[3]) 和右边 (pts[1]→pts[2]) 的平均作为高度
    w_top = float(np.linalg.norm(pts_src[1] - pts_src[0]))
    w_bot = float(np.linalg.norm(pts_src[2] - pts_src[3]))
    w_src = (w_top + w_bot) / 2.0

    h_left = float(np.linalg.norm(pts_src[3] - pts_src[0]))
    h_right = float(np.linalg.norm(pts_src[2] - pts_src[1]))
    h_src = (h_left + h_right) / 2.0

    # ---- 3. 验证源四边形的像素宽高比 ----
    # 若高度太小（<1px），宽高比无意义，直接判定为无效
    if h_src < 1.0:
        actual_ratio = float("inf")
        is_valid = False
    else:
        actual_ratio = w_src / h_src
        # 计算相对误差: |实际比 - 真实比| / 真实比
        ratio_error = abs(actual_ratio - real_aspect_ratio) / real_aspect_ratio
        is_valid = ratio_error <= tolerance

    # ---- 4. 确定目标尺寸 ----
    # 默认保持源四边形宽度不变，高度由真实宽高比反推
    if target_width is None:
        target_width = max(int(w_src), 1)

    target_height = max(int(target_width / real_aspect_ratio), 1)

    # ---- 5. 透视变换 ----
    # 目标矩形的四个角点按与源点相同的顺时针顺序排列
    pts_dst = np.array(
        [
            [0, 0],  # 左上
            [target_width - 1, 0],  # 右上
            [target_width - 1, target_height - 1],  # 右下
            [0, target_height - 1],  # 左下
        ],
        dtype=np.float32,
    )

    # 计算 3×3 透视变换矩阵，然后执行变换
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, M, (target_width, target_height))

    return warped, is_valid, actual_ratio


# ============================================================================
# 检测函数
# ============================================================================


def detect_rect(
    img: MatLike,
    gray: MatLike,
    min_area: np.uint32,
    white_area: np.uint32,
    real_aspect_ratio: float,
    target_width: Optional[int] = None,
    tolerance: float = 0.1,
    epsilon: float = 0.02,
    reject_status: Optional[Dict[str, int]] = None,
) -> MatLike | None:
    """从二值边缘图像中检测最佳矩形区域。

    此函数实现了一个多级筛选管道，逐步滤除不符合条件的轮廓：

        [轮廓] → 面积过滤 → 四边形逼近 → 白色区域亮度 → 宽高比验证 → [最佳矩形]

    筛选层级:
        1. 面积过滤     — 丢弃面积小于 min_area 的轮廓（噪声点/碎片）
        2. 四边形逼近   — 用 approxPolyDP 逼近，丢弃非四边形的轮廓
        3. 白色区域亮度 — 在轮廓掩膜内计算灰度均值，丢弃过暗的区域
        4. 宽高比验证   — 直接计算四边形对边长度，避免无用透视变换

    参数:
        img               : 二值边缘图像（通常是 Canny 输出），用于 findContours
        gray              : 原始灰度图，用于计算区域内平均亮度
        min_area          : 最小轮廓面积阈值，小于此值的轮廓被丢弃
        white_area        : 最小平均灰度值阈值（0–255），用于判断区域是否"够白"
        real_aspect_ratio : 目标矩形的真实物理宽高比
        target_width      : 为兼容旧调用保留；矩形筛选阶段不再生成透视图
        tolerance         : 宽高比允许的相对误差
        epsilon           : approxPolyDP 的逼近精度因子，
                            实际 epsilon = epsilon × 轮廓周长
                            默认 0.02 表示用周长的 2% 作为逼近精度
        reject_status     : 可变的拒绝统计字典，记录各级过滤的剔除数量
                            键: "area", "quad", "white_region", "aspect_ratio"

    返回:
        best_rect: 形状 (4, 2) 的 int32 numpy 数组，
                   通过所有筛选的轮廓中面积最大的那一个的四个角点
                   注意：若有多个候选，取面积最大者；仅有一个候选时直接返回
    """
    if reject_status is None:
        reject_status = {
            "area": 0,
            "quad": 0,
            "white_region": 0,
            "aspect_ratio": 0,
        }

    rects: List = []

    # 在二值边缘图中查找所有轮廓
    # RETR_LIST — 不建立层级关系，返回所有轮廓（性能最优）
    # CHAIN_APPROX_SIMPLE — 仅保留拐点，压缩水平/垂直/对角线段
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # ---- 第一级：面积过滤 ----
        area = cv2.contourArea(cnt)
        if area < min_area:
            reject_status["area"] += 1
            continue

        # ---- 第二级：四边形逼近 ----
        # 用指定精度逼近轮廓的多边形，检查是否为四边形
        peri = cv2.arcLength(cnt, True)  # True = 闭合轮廓
        approx = cv2.approxPolyDP(cnt, epsilon * peri, True)
        if len(approx) != 4:
            reject_status["quad"] += 1
            continue

        # ---- 第三级：白色区域亮度检查 ----
        # 思路：目标矩形区域（如车牌底色）应该是亮白色
        # 在轮廓内部创建掩膜，计算该区域的平均灰度值
        x, y, w, h = cv2.boundingRect(approx)
        roi_gray = gray[y : y + h, x : x + w]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        shifted = approx - np.array([[[x, y]]], dtype=approx.dtype)
        cv2.drawContours(roi_mask, [shifted], -1, 255, -1)
        mean_val = cv2.mean(roi_gray, mask=roi_mask)[0]
        if mean_val < white_area:
            reject_status["white_region"] += 1
            continue

        # ---- 第四级：宽高比验证 ----
        pts = approx.reshape(4, 2)
        ordered = order_points(pts).astype(np.float32)
        w_src = (
            np.linalg.norm(ordered[1] - ordered[0])
            + np.linalg.norm(ordered[2] - ordered[3])
        ) / 2.0
        h_src = (
            np.linalg.norm(ordered[3] - ordered[0])
            + np.linalg.norm(ordered[2] - ordered[1])
        ) / 2.0
        actual_ratio = float("inf") if h_src < 1.0 else float(w_src / h_src)
        is_valid = (
            h_src >= 1.0
            and abs(actual_ratio - real_aspect_ratio) / real_aspect_ratio <= tolerance
        )

        if not is_valid:
            reject_status["aspect_ratio"] += 1
            continue

        # 通过所有筛选，加入候选列表
        rects.append(pts)

    # 在候选矩形中选取面积最大的作为最终结果
    # 若无候选矩形，返回 None，由调用方处理
    if not rects:
        return None
    best_rect = order_points(max(rects, key=cv2.contourArea))
    return best_rect


def detect_laser_mask(
    img: MatLike,
    min_area: np.uint32,
) -> MatLike | None:
    """从二值图像中检测最大的激光遮罩轮廓。

    与 detect_rect 相比，此函数更简单：仅按面积过滤，然后返回最大轮廓。
    激光遮罩通常形状不规则，因此不做四边形逼近或宽高比验证。

    处理流程:
        1. 在二值图中查找外轮廓（RETR_EXTERNAL，仅最外层）
        2. 过滤掉面积小于 min_area 的噪声轮廓
        3. 返回剩余轮廓中面积最大的

    参数:
        img      : 二值图像（白色前景 + 黑色背景），仅检测外轮廓
        min_area : 最小轮廓面积阈值

    返回:
        best_lm: 面积最大的轮廓点集，形状 (N, 1, 2)
    """
    lms: List = []

    # RETR_EXTERNAL — 仅检索最外层轮廓，忽略嵌套的内部轮廓
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        lms.append(cnt)

    # 若无符合条件的轮廓，返回 None，由调用方处理
    if not lms:
        return None
    best_lm = max(lms, key=cv2.contourArea)
    return best_lm


# ============================================================================
# 预处理函数
# ============================================================================


def preprocess(
    frame: MatLike,
    kernel: MatLike,
    rect_lab_thresholds: Tuple[Sequence[int], Sequence[int]],
    laser_mask_thresholds: Tuple[Sequence[int], Sequence[int]],
    canny_thresholds: Tuple[int, int] = (50, 150),
) -> Tuple[MatLike, MatLike, MatLike]:
    """对输入帧进行预处理，生成用于矩形检测和激光遮罩检测的二值图像。

    此函数是检测管线的入口，在一次遍历中同时产出两条处理链路的结果：
        - 矩形检测链: LAB → inRange → GaussianBlur → Canny → morphologyEx
        - 激光遮罩链: LAB → inRange → GaussianBlur → morphologyEx

    两条链共享一次颜色空间转换（BGR → LAB + GRAY），避免了重复计算。

    预处理步骤详解:

        **矩形检测链 (rect_edges)**:
        1. BGR → LAB 色彩空间转换
        2. inRange 颜色阈值过滤，提取目标颜色区域
        3. GaussianBlur 高斯模糊，抑制噪点
        4. Canny 边缘检测，提取区域边缘
        5. morphologyEx(MORPH_CLOSE) 闭运算，闭合边缘断裂

        **激光遮罩链 (lm_binary)**:
        1. BGR → LAB（与矩形链共用）
        2. inRange 颜色阈值过滤
        3. GaussianBlur 高斯模糊
        4. morphologyEx(MORPH_CLOSE) 闭运算，填充遮罩内部空洞

    参数:
        frame                 : 输入帧（BGR 格式，uint8）
        kernel                : 形态学操作的结构元素（矩形/椭圆/十字形），
                                同时用于 GaussianBlur 的核大小
                                核尺寸由 kernel.shape 决定
        rect_lab_thresholds   : 矩形检测的 LAB 阈值 (lower, upper)，
                                应为 OpenCV 格式（已通过 cvt_mvlab2cv 转换）
        laser_mask_thresholds : 激光遮罩检测的 LAB 阈值 (lower, upper)
        canny_thresholds      : Canny 边缘检测的阈值 (low, high)，
                                默认 [50, 150]。注意：代码中实际使用
                                canny_thresholds[0] 作为两个阈值

    返回:
        (rect_edges, lm_binary, gray):
            - rect_edges : 矩形检测用的二值边缘图
            - lm_binary  : 激光遮罩检测用的二值图
            - gray       : 原始灰度图，供后续 detect_rect 使用
    """
    # ---- 色彩空间转换（两条链共用，只做一次） ----
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # BGR → CIELAB
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # BGR → 灰度

    # 获取核尺寸：GaussianBlur 要求核宽高均为正奇数
    height, width = kernel.shape
    if height <= 0 or width <= 0 or height % 2 == 0 or width % 2 == 0:
        raise ValueError(f"kernel 尺寸必须为正奇数，实际为 {(width, height)}")

    # ============ 矩形检测预处理链 ============

    # 步骤 1: LAB 颜色空间阈值过滤
    # inRange 保留落在 [lower, upper] 范围内的像素（设为 255），其余置 0
    rect_binary = cv2.inRange(lab, rect_lab_thresholds[0], rect_lab_thresholds[1])

    # 步骤 2: 高斯模糊 — 抑制高频噪点，使边缘检测更稳健
    # 核大小取自 kernel.shape，sigmaX=0 表示由核大小自动计算
    rect_blurred = cv2.GaussianBlur(rect_binary, (width, height), 0)

    # 步骤 3: Canny 边缘检测
    # low=high=canny_thresholds[0] — 使用单阈值，只保留强边缘
    rect_edges = cv2.Canny(rect_blurred, canny_thresholds[0], canny_thresholds[1])

    # 步骤 4: 形态学闭运算（先膨胀后腐蚀）
    # 目的：闭合 Canny 检测到的边缘断裂，形成连续轮廓
    rect_edges = cv2.morphologyEx(rect_edges, cv2.MORPH_CLOSE, kernel)

    # ============ 激光遮罩预处理链 ============

    # 步骤 1: LAB 颜色空间阈值过滤
    lm_binary = cv2.inRange(lab, laser_mask_thresholds[0], laser_mask_thresholds[1])

    # 步骤 2: 开运算
    # lm_binary = cv2.morphologyEx(lm_binary, cv2.MORPH_OPEN, kernel)
    #
    # # 步骤 3: 形态学闭运算 — 填充遮罩内部的细小空洞
    # lm_binary = cv2.morphologyEx(lm_binary, cv2.MORPH_CLOSE, kernel)

    return rect_edges, lm_binary, gray
