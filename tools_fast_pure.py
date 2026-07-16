"""
工具加速模块 — 纯 Python 优化（零依赖，无编译）
优化手段：
  1. mask 缓存预分配，避免 per-contour np.zeros_like
  2. 轮廓分片并行（native OpenCV 调用释放 GIL，大轮廓列表时可加速 2×）
"""

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tools.tools import perspective_correct_and_validate, order_points


class _RectContext:
    """矩形检测上下文 — 预分配缓存，避免每帧重复 malloc"""
    __slots__ = ("mask", "white_buf")

    def __init__(self, h, w):
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self.white_buf = np.empty((h, w), dtype=np.uint8)

    def reset_mask(self):
        self.mask.fill(0)


def _process_chunk(contours, ctx, gray, min_area, white_area,
                   real_aspect_ratio, target_width, tolerance, epsilon,
                   reject_status, start_idx):
    """处理一批轮廓 — 被并行调用，各自拥有本地 rects 列表"""
    rects = []
    local_rejects = {"area": 0, "quad": 0, "white_region": 0, "aspect_ratio": 0}

    for cnt in contours:
        area_val = cv2.contourArea(cnt)
        if area_val < min_area:
            local_rejects["area"] += 1
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon * peri, True)
        if len(approx) != 4:
            local_rejects["quad"] += 1
            continue

        # 白色区域（复用 ctx.mask）
        ctx.reset_mask()
        cv2.drawContours(ctx.mask, [approx], -1, 255, -1)
        cv2.bitwise_and(gray, gray, dst=ctx.white_buf, mask=ctx.mask)
        mean_val = cv2.mean(ctx.white_buf, mask=ctx.mask)[0]
        if mean_val < white_area:
            local_rejects["white_region"] += 1
            continue

        pts = approx.reshape(4, 2)
        _, is_valid, _ = perspective_correct_and_validate(
            gray, pts, real_aspect_ratio,
            target_width=target_width, tolerance=tolerance)
        if not is_valid:
            local_rejects["aspect_ratio"] += 1
            continue

        rects.append(pts)

    return rects, local_rejects


def detect_rect_fast(
    img, gray,
    min_area, white_area,
    real_aspect_ratio,
    target_width=None,
    tolerance=0.1,
    epsilon=0.02,
    reject_status=None,
    ctx=None,
    num_workers=0,
):
    """加速版 detect_rect

    参数:
        num_workers: 轮廓并行处理数。0=自动(轮廓>20时用2核)，1=纯串行
        ctx: 外部传入的 _RectContext 缓存池，不传则内部自动创建
        其余参数同 tools.tools.detect_rect
    """
    if reject_status is None:
        reject_status = {"area": 0, "quad": 0, "white_region": 0, "aspect_ratio": 0}

    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)

    if n == 0:
        return None

    h, w = gray.shape

    # 决定并行度
    if num_workers == 0:
        num_workers = 2 if n > 20 else 1

    if num_workers <= 1:
        # 串行路径 — 用本地缓存
        ctx_local = ctx or _RectContext(h, w)
        rects = []
        for cnt in contours:
            area_val = cv2.contourArea(cnt)
            if area_val < min_area:
                reject_status["area"] += 1
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon * peri, True)
            if len(approx) != 4:
                reject_status["quad"] += 1
                continue

            ctx_local.reset_mask()
            cv2.drawContours(ctx_local.mask, [approx], -1, 255, -1)
            cv2.bitwise_and(gray, gray, dst=ctx_local.white_buf, mask=ctx_local.mask)
            mean_val = cv2.mean(ctx_local.white_buf, mask=ctx_local.mask)[0]
            if mean_val < white_area:
                reject_status["white_region"] += 1
                continue

            pts = approx.reshape(4, 2)
            _, is_valid, _ = perspective_correct_and_validate(
                gray, pts, real_aspect_ratio,
                target_width=target_width, tolerance=tolerance)
            if not is_valid:
                reject_status["aspect_ratio"] += 1
                continue

            rects.append(pts)
    else:
        # 并行路径 — 分片
        num_workers = min(num_workers, n, 4)
        chunks = []
        base_size = n // num_workers
        remainder = n % num_workers
        idx = 0
        for i in range(num_workers):
            size = base_size + (1 if i < remainder else 0)
            chunks.append(contours[idx:idx + size])
            idx += size

        # 每个 chunk 用独立 context（共享 mask 会竞争）
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = []
            for chunk in chunks:
                c = _RectContext(h, w)
                f = ex.submit(
                    _process_chunk, chunk, c, gray,
                    min_area, white_area,
                    real_aspect_ratio, target_width, tolerance, epsilon,
                    reject_status, 0,
                )
                futures.append(f)

            all_rects = []
            for f in futures:
                r, lr = f.result()
                all_rects.extend(r)
                for k in ("area", "quad", "white_region", "aspect_ratio"):
                    reject_status[k] += lr[k]

            rects = all_rects

    if not rects:
        return None
    best = max(rects, key=cv2.contourArea)
    return order_points(best)


def detect_laser_mask_fast(img, min_area):
    """加速版激光遮罩检测（基本同原版，for 循环微调）"""
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            rects.append(cnt)
    if not rects:
        return None
    return max(rects, key=cv2.contourArea)
