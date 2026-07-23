"""相机捕获 + 矩形追踪 + 激光点追踪 — Web 推流版。"""

import cv2
import numpy as np

from tools._threshold import Binarizer
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
DEFAULT_LAB_THRESHOLDS = [12, 100, -53, 7, -38, 31]

MIN_AREA = 2000
MIN_WHITE = 10
KERNEL_SIZE = 5

# ============================================================================
# 初始化
# ============================================================================

config = PipelineConfig(
    source="usb",
    device="/dev/video0",
    width=640,
    height=480,
    fps=120,
    flip_method=6,
)
camera = JetsonCamera(config)
if not camera.open():
    raise RuntimeError(f"无法打开相机：{camera.last_error}")
fps = FpsShow()
binarizer = Binarizer(strategy="range")
rect_tracker = RectTracker(track_radius=250, smooth_alpha=0.6)
laser_detector = LaserSpotDetector(
    track_radius=120,
    smooth_alpha=0.65,
    full_search_interval=30,
    min_area=5,
    max_area=1000,
    morph_kernel_size=3,
    roi_margin=4,
    max_aspect_ratio=3.0,
    min_confidence=0.25,
    color_mode="blue",
    min_color_excess=40,
    min_color_value=80,
    threshold=[99, 100, -32, 28, -38, 26],
)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_SIZE, KERNEL_SIZE))
rect_lab_lower, rect_lab_upper = cvt_mvlab2cv(DEFAULT_LAB_THRESHOLDS)

server = DebugServer(port=8080)
server.start()
best_rect = None
# ============================================================================
# 主循环
# ============================================================================

consecutive_read_failures = 0
camera_error = None
while True:
    ok, frame = camera.read()
    if not ok or frame is None:
        consecutive_read_failures += 1
        if consecutive_read_failures >= 30:
            camera_error = "相机连续读取失败 30 次"
            break
        continue
    consecutive_read_failures = 0
    # 如果矩形没有找到，那就进行全图学习
    if best_rect is not None:
        binarizer.learn(
            frame=frame,
            roi_x=0,
            roi_y=0,
            roi_w=cv2.get(cv2.CAP_PROP_FRAME_WIDGH),
            roi_h=cv2.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )
    if binarizer.is_learned:
        param = binarizer.params()
        rect_lab_lower = param["lower"]
        rect_lab_upper = param["upper"]

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

    # # ── 激光点追踪 ──
    # spot = (
    #     laser_detector.detect(frame, search_polygon=best_rect)
    #     if best_rect is not None
    #     else None
    # )
    # if graph is not None:
    #     graph.draw_border(frame)
    #     graph.draw_corners(frame)
    #
    # if spot is not None:
    #     if graph is not None:
    #         laser_pt = np.array([[spot.x, spot.y]], dtype=np.float32)
    #         plane_pt = graph.map_from_image(laser_pt)[0]
    #         u = plane_pt[0] / max(graph.plane_w - 1, 1)
    #         v = plane_pt[1] / max(graph.plane_h - 1, 1)
    #         if 0 <= u <= 1 and 0 <= v <= 1:
    #             graph.draw_point(frame, u, v)
    #             graph.draw_cross(frame, u, v)
    #             graph.draw_label(frame, u, v, f"conf:{spot.confidence:.2f}")

    cv2.putText(
        frame,
        f"area:{reject_status['area']},quad:{reject_status['quad']},aspect_ratio:{reject_status['aspect_ratio']},hite_region:{reject_status['white_region']}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
    )

    # ── 推流 ──
    fps.show(frame)
    server.update_frame(0, frame)

    server.update_frame(1, rect_edges)
camera.release()
server.stop()
cv2.destroyAllWindows()
if camera_error is not None:
    raise RuntimeError(camera_error)
