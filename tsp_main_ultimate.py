"""去畸变 → 裁剪黑边 → 检测 → 原图画框。
终极优化版：降采样 + 双线程 + 轮廓并行 + mask 缓存 + 核分配
"""

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tools.web import DebugServer, CameraManager, ParamRegistry
from tools.tools import (
    FpsShow,
    cvt_mvlab2cv,
    enable_opencl,
    preprocess,
)
from tools_fast_pure import detect_rect_fast, detect_laser_mask_fast

# ---- 目标真实宽高比 ----
REAL_ASPECT_RATIO = 0.657
ASPECT_TOLERANCE = 0.4

# ---- 激光遮罩 LAB 阈值 ----
LASER_MV_LABVALUE = [31, 58, 11, 63, -19, 20]

# ---- 处理降采样 ----
PROCESS_SCALE = 0.5


# 每个工作线程只让 OpenCV 用 1 核
def _thread_init():
    cv2.setNumThreads(1)


def main():
    enable_opencl()
    cv2.setNumThreads(2)  # 主线程的 preprocess 用 2 核

    cam = CameraManager()
    cap = cam.open(index=10, width=1280, height=720)

    params = ParamRegistry()
    params.add("kernel", type=int, default=5, range=(1, 21), step=2, group="形态学")
    params.add("min_area", type=int, default=2000, range=(100, 20000), group="筛选")
    params.add("min_white", type=int, default=60, range=(0, 255), group="筛选")
    params.add("lm_min_area", type=int, default=200, range=(50, 10000), group="激光")

    server = DebugServer(params=params, port=8080)
    server.start()
    fps = FpsShow()

    with np.load("param.npz") as p:
        mtx = p["mtx"]
        dist = p["dist"]

    rect_lab_lower, rect_lab_upper = cvt_mvlab2cv()
    lm_lab_lower, lm_lab_upper = cvt_mvlab2cv(LASER_MV_LABVALUE)

    executor = ThreadPoolExecutor(max_workers=2, initializer=_thread_init)
    rect_future = None
    laser_future = None
    inv_scale = 1.0 / PROCESS_SCALE

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 0)
        frame1 = frame.copy()

        ksize = params.get("kernel")
        min_area = params.get("min_area")
        min_white = params.get("min_white")
        lm_min_area = params.get("lm_min_area")
        kernel = np.ones((ksize, ksize), dtype=np.uint8)

        # ---- 降采样 ----
        frame_u = cv2.UMat(frame)
        small = cv2.resize(
            frame_u, None,
            fx=PROCESS_SCALE, fy=PROCESS_SCALE,
            interpolation=cv2.INTER_LINEAR,
        )

        rect_edges, lm_binary, gray = preprocess(
            small, kernel,
            [rect_lab_lower, rect_lab_upper],
            [lm_lab_lower, lm_lab_upper],
        )

        # ---- 取前次异步结果（可视化） ----
        best_rect = rect_future.result() if rect_future is not None else None
        reject_status = {"area": 0, "quad": 0, "white_region": 0, "aspect_ratio": 0}

        if best_rect is not None:
            scaled_rect = (best_rect * inv_scale).astype(np.int32)
            rect_center_x = int(scaled_rect[:, 0].mean())
            rect_center_y = int(scaled_rect[:, 1].mean())
            cv2.polylines(frame, [scaled_rect], True, (0, 255, 0), 2)
            cv2.circle(frame, (rect_center_x, rect_center_y), 5, (0, 255, 0), -1)
        else:
            rect_center_x = rect_center_y = None

        best_lm = laser_future.result() if laser_future is not None else None

        if best_lm is not None:
            lmx, lmy, lmw, lmh = cv2.boundingRect(best_lm)
            lm_center_x = int((lmx + lmw // 2) * inv_scale)
            lm_center_y = int((lmy + lmh // 2) * inv_scale)
            cv2.rectangle(
                frame,
                (int(lmx * inv_scale), int(lmy * inv_scale)),
                (int((lmx + lmw) * inv_scale), int((lmy + lmh) * inv_scale)),
                (255, 0, 0), 2,
            )
            cv2.circle(frame, (lm_center_x, lm_center_y), 5, (255, 0, 0), -1)
        else:
            lm_center_x = lm_center_y = None

        # ---- 提交异步检测（Cython 加速版） ----
        rect_future = executor.submit(
            detect_rect_fast, rect_edges, gray, min_area, min_white,
            REAL_ASPECT_RATIO, None, ASPECT_TOLERANCE, 0.02, reject_status,
        )
        laser_future = executor.submit(
            detect_laser_mask_fast, lm_binary, lm_min_area,
        )

        # ---- 叠加信息 ----
        if rect_center_x is not None:
            cv2.putText(frame, f"rect:({rect_center_x},{rect_center_y})",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if lm_center_x is not None:
            cv2.putText(frame, f"laser:({lm_center_x},{lm_center_y})",
                        (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        fps.show(frame)

        cv2.putText(
            frame,
            f"area:{reject_status['area']} asp:{reject_status['aspect_ratio']} "
            f"quad:{reject_status['quad']} white:{reject_status['white_region']}",
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )

        server.update_frame(0, frame1)
        server.update_frame(1, frame)


if __name__ == "__main__":
    main()
