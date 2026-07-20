"""矩形检测 — 多级筛选管道（面积→四边形→亮度→宽高比）+ 多线程并行。"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ._geometry import order_points, perspective_correct_and_validate

MatLike = np.ndarray


def _process_contour_chunk(
    contours_chunk: List,
    gray: MatLike,
    min_area: float,
    white_area: float,
    real_aspect_ratio: float,
    target_width: Optional[int],
    tolerance: float,
    epsilon: float,
) -> Tuple[List[np.ndarray], Dict[str, int]]:
    """处理一组轮廓（供多线程并行调用），返回 (候选矩形列表, 拒绝计数)。"""
    rects: List[np.ndarray] = []
    reject = {"area": 0, "quad": 0, "white_region": 0, "aspect_ratio": 0}

    for cnt in contours_chunk:
        area = cv2.contourArea(cnt)
        if area < min_area:
            reject["area"] += 1
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon * peri, True)
        if len(approx) != 4:
            reject["quad"] += 1
            continue

        mask_poly = np.zeros_like(gray)
        cv2.drawContours(mask_poly, [approx], -1, 255, -1)
        white_region = cv2.bitwise_and(gray, gray, mask=mask_poly)
        mean_val = cv2.mean(white_region, mask=mask_poly)[0]
        if mean_val < white_area:
            reject["white_region"] += 1
            continue

        pts = approx.reshape(4, 2)
        _, is_valid, _ = perspective_correct_and_validate(
            gray, pts, real_aspect_ratio, target_width=target_width, tolerance=tolerance
        )
        if not is_valid:
            reject["aspect_ratio"] += 1
            continue

        rects.append(pts)

    return rects, reject


def detect_rect(
    img: MatLike,
    gray: MatLike,
    min_area: np.uint32,
    white_area: np.uint32,
    real_aspect_ratio: float,
    target_width: Optional[int] = None,
    tolerance: float = 0.1,
    epsilon: float = 0.02,
    reject_status: Dict = {"area": 0, "quad": 0, "white_region": 0, "aspect_ratio": 0},
    num_workers: int = 0,
) -> Optional[MatLike]:
    """从二值边缘图像中检测最佳矩形区域。"""
    rects: List[np.ndarray] = []
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if num_workers == 0:
        num_workers = min(os.cpu_count() or 4, 4)
    use_parallel = num_workers > 1 and len(contours) >= 50

    if use_parallel:
        chunk_size = max(1, len(contours) // num_workers)
        chunks: List[List] = []
        for i in range(num_workers):
            start = i * chunk_size
            end = start + chunk_size if i < num_workers - 1 else len(contours)
            if start >= end:
                break
            chunks.append(contours[start:end])

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(
                    _process_contour_chunk,
                    chunk, gray,
                    float(min_area), float(white_area),
                    real_aspect_ratio, target_width, tolerance, epsilon,
                )
                for chunk in chunks
            ]
            for f in as_completed(futures):
                chunk_rects, chunk_reject = f.result()
                rects.extend(chunk_rects)
                for k in reject_status:
                    reject_status[k] += chunk_reject[k]
    else:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                reject_status["area"] += 1
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon * peri, True)
            if len(approx) != 4:
                reject_status["quad"] += 1
                continue

            mask_poly = np.zeros_like(gray)
            cv2.drawContours(mask_poly, [approx], -1, 255, -1)
            white_region = cv2.bitwise_and(gray, gray, mask=mask_poly)
            mean_val = cv2.mean(white_region, mask=mask_poly)[0]
            if mean_val < white_area:
                reject_status["white_region"] += 1
                continue

            pts = approx.reshape(4, 2)
            _, is_valid, _ = perspective_correct_and_validate(
                gray, pts, real_aspect_ratio,
                target_width=target_width, tolerance=tolerance,
            )
            if not is_valid:
                reject_status["aspect_ratio"] += 1
                continue

            rects.append(pts)

    if not rects:
        return None
    best_rect = order_points(max(rects, key=cv2.contourArea))
    return best_rect
