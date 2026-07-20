"""相机捕获 + 矩形追踪 + 激光点追踪 — Web 推流版。"""

import cv2
import numpy as np

from tools.hardware_pipeline import JetsonCamera, PipelineConfig
from tools.tools import (
    DrawGraph,
    FpsShow,
    LaserSpotDetector,
    RectTracker,
    cvt_mvlab2cv,
    preprocess,
)
from tools.web import DebugServer

# ============================================================================
# 参数
# ============================================================================

REAL_ASPECT_RATIO = 0.657
ASPECT_TOLERANCE = 0.4
DEFAULT_LAB_THRESHOLDS = [41, 74, -14, 13, -27, 31]

MIN_AREA = 2000
MIN_WHITE = 60
KERNEL_SIZE = 5

# ============================================================================
# 初始化
# ============================================================================

config = PipelineConfig(
    source="usb",
    device="/dev/video0",
    width=640,
    height=480,
    fps=60,
    flip_method=6,
)
camera = JetsonCamera(config)
camera.open()

fps = FpsShow()
rect_tracker = RectTracker(track_radius=250, smooth_alpha=0.6)
laser_detector = LaserSpotDetector(
    track_radius=120,
    smooth_alpha=0.65,
    full_search_interval=30,
    min_area=10,
)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_SIZE, KERNEL_SIZE))
rect_lab_lower, rect_lab_upper = cvt_mvlab2cv(DEFAULT_LAB_THRESHOLDS)

server = DebugServer(port=8080)
server.start()

# ============================================================================
# 主循环
# ============================================================================

while True:
    ok, frame = camera.read()
    if not ok or frame is None:
        continue

    rect_edges, gray = preprocess(frame, kernel, (rect_lab_lower, rect_lab_upper))

    # ── 矩形追踪 ──
    reject_status = {"area": 0, "quad": 0, "white_region": 0, "aspect_ratio": 0}
    best_rect = rect_tracker.track(
        rect_edges,
        gray,
        MIN_AREA,
        MIN_WHITE,
        REAL_ASPECT_RATIO,
        tolerance=ASPECT_TOLERANCE,
        reject_status=reject_status,
    )

    graph = None
    if best_rect is not None:
        w_top = np.linalg.norm(best_rect[1] - best_rect[0])
        w_bot = np.linalg.norm(best_rect[2] - best_rect[3])
        h_left = np.linalg.norm(best_rect[3] - best_rect[0])
        h_right = np.linalg.norm(best_rect[2] - best_rect[1])
        plane_w = max(int((w_top + w_bot) / 2), 1)
        plane_h = max(int((h_left + h_right) / 2), 1)

        graph = DrawGraph(best_rect.astype(np.float32), plane_w, plane_h)
        graph.draw_border(frame)
        graph.draw_corners(frame)
        center_xy = best_rect.mean(axis=0)

        print(f"rect:{center_xy[0]},{center_xy[1]}")

    # ── 激光点追踪 ──
    spot = laser_detector.detect(frame)
    if spot is not None:
        if graph is not None:
            laser_pt = np.array([[spot.x, spot.y]], dtype=np.float32)
            plane_pt = graph.map_from_image(laser_pt)[0]
            u = plane_pt[0] / (graph.plane_w - 1)
            v = plane_pt[1] / (graph.plane_h - 1)
            if 0 <= u <= 1 and 0 <= v <= 1:
                graph.draw_point(frame, u, v)
                graph.draw_cross(frame, u, v)
        else:
            ix, iy = int(spot.x), int(spot.y)
            cv2.drawMarker(frame, (ix, iy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(frame, (ix, iy), 8, (0, 255, 0), 1)

    # ── 推流 ──
    print(f"lm:{ix},{iy}")
    fps.show(frame)
    server.update_frame(0, frame)
    server.update_frame(1, cv2.cvtColor(rect_edges, cv2.COLOR_GRAY2BGR))

camera.release()
server.stop()
cv2.destroyAllWindows()
