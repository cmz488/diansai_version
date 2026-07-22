"""矩形检测 — Binarizer 两阶段自动学习测试。

阶段 1 (adaptive):  Binarizer(adaptive) → Canny → RectTracker → 检测矩形
阶段 2 (range):    RectTracker 检出矩形 → 自动学习 LAB 范围 → 精确分割

流程: adaptive 粗定位 → 自动提取矩形 ROI → range 学习 → 精检测

用法::

    python tests/test_rect_detect.py

操作:
    Trackbar                        调 Canny / min_area / white_area
    q / ESC                         退出
    r                               重置 range 学习
    s                               截图
    a                               手动触发学习（从当前 adaptive 矩形）

画面 (2行 × 4列):
    行1: 原图+检测框 | adaptive二值图 | adaptive边缘图 | adaptive矩形
    行2: 原图+检测框 | range二值图   | range边缘图   | range矩形
"""

import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ.pop("WAYLAND_DISPLAY", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from tools._threshold import Binarizer
from tools._rect_detect import detect_rect
from tools._tracking import RectTracker

# ============================================================================

WIN_NAME = "Rect Detect | Auto-Learn"
REAL_ASPECT_RATIO = 0.657
ASPECT_TOLERANCE = 0.4
LEARN_COOLDOWN = 3  # 学习冷却帧数（无矩形时每 3 帧学一次）
LEARN_STABILITY = 2  # 连续稳定帧数（降低门槛，更快触发）

# 学习去重：新旧 LAB 范围偏差小于此阈值则跳过
DEDUP_L_THRESHOLD = 10
DEDUP_AB_THRESHOLD = 5


def nothing(_: int) -> None:
    pass


def pipeline_adaptive(
    frame: np.ndarray,
    canny_lo: int,
    canny_hi: int,
    kernel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive 管线: Binarizer → Canny → 闭运算。"""
    b = Binarizer(strategy="adaptive")
    mask = b.apply(frame)
    edges = cv2.Canny(mask, canny_lo, canny_hi)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return mask, edges


def pipeline_range(
    frame: np.ndarray,
    binarizer: Binarizer,
    canny_lo: int,
    canny_hi: int,
    kernel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Range 管线: 已学习 Binarizer → Canny → 闭运算。"""
    if not binarizer.is_learned:
        h, w = frame.shape[:2]
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
    mask = binarizer.apply(frame)
    edges = cv2.Canny(mask, canny_lo, canny_hi)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return mask, edges


def draw_rect(
    img: np.ndarray,
    pts: np.ndarray | None,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    out = img.copy()
    if pts is not None:
        pts_i = pts.astype(np.int32)
        cv2.polylines(out, [pts_i], True, color, 2)
        for i, pt in enumerate(pts_i):
            cv2.circle(out, tuple(pt), 4, color, -1)
    return out


def main() -> None:
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("无法打开摄像头 (device 0)")
        sys.exit(1)

    ok, frame = cap.read()
    if not ok:
        print("无法读取摄像头帧")
        sys.exit(1)

    fh, fw = frame.shape[:2]
    cell_w = 320
    scale = cell_w / fw
    cell_h = int(fh * scale)

    print(f"摄像头: {fw}×{fh}")
    print("流程: adaptive 粗定位 → 自动学习 ROI → range 精检测")
    print("按键: q/ESC=退出  r=重置  s=截图  a=手动触发学习")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, cell_w * 4, cell_h * 2 + 80)
    cv2.imshow(WIN_NAME, np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.waitKey(100)

    cv2.createTrackbar("canny_lo", WIN_NAME, 50, 255, nothing)
    cv2.createTrackbar("canny_hi", WIN_NAME, 150, 255, nothing)
    cv2.createTrackbar("min_area", WIN_NAME, 20, 200, nothing)
    cv2.createTrackbar("white_area", WIN_NAME, 10, 255, nothing)

    # 两个独立的追踪器
    tracker_a = RectTracker(track_radius=250, smooth_alpha=0.6, full_search_interval=30)
    tracker_r = RectTracker(track_radius=250, smooth_alpha=0.6, full_search_interval=30)

    range_bin = Binarizer(strategy="range", min_range=8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 自动学习状态
    learn_cooldown = 0
    stable_count = 0
    last_learned_rect: np.ndarray | None = None

    def _lab_similar(lo_new, up_new, lo_old, up_old) -> bool:
        """判断新旧 LAB 范围是否相似（跳过重复学习）。"""
        if lo_old is None:
            return False
        l_diff = max(abs(int(lo_new[0]) - int(lo_old[0])),
                     abs(int(up_new[0]) - int(up_old[0])))
        a_diff = max(abs(int(lo_new[1]) - int(lo_old[1])),
                     abs(int(up_new[1]) - int(up_old[1])))
        b_diff = max(abs(int(lo_new[2]) - int(lo_old[2])),
                     abs(int(up_new[2]) - int(up_old[2])))
        return (l_diff < DEDUP_L_THRESHOLD and
                a_diff < DEDUP_AB_THRESHOLD and
                b_diff < DEDUP_AB_THRESHOLD)

    def _do_learn(frame_src, roi_x, roi_y, roi_w, roi_h, source_label):
        """执行学习，带去重检查。"""
        old_lo = range_bin._lower
        old_up = range_bin._upper
        try:
            range_bin.learn(frame_src, roi_x=roi_x, roi_y=roi_y,
                            roi_w=roi_w, roi_h=roi_h)
        except RuntimeError as e:
            print(f"[学习失败-{source_label}] {e}")
            return
        if _lab_similar(range_bin._lower, range_bin._upper, old_lo, old_up):
            # 恢复旧值，避免无意义更新
            range_bin._lower = old_lo
            range_bin._upper = old_up
            return
        print(f"[学习-{source_label}] {range_bin.describe()}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        frame_small = cv2.resize(frame, (cell_w, cell_h))
        gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        c_lo = cv2.getTrackbarPos("canny_lo", WIN_NAME)
        c_hi = cv2.getTrackbarPos("canny_hi", WIN_NAME)
        min_a = cv2.getTrackbarPos("min_area", WIN_NAME) * 100
        white_a = cv2.getTrackbarPos("white_area", WIN_NAME)

        # ---- 管线 A: adaptive + RectTracker ----
        mask_a, edges_a = pipeline_adaptive(frame_small, c_lo, c_hi, kernel)
        reject_a: dict = {"area": 0, "quad": 0, "white_region": 0, "aspect_ratio": 0}
        rect_a = tracker_a.track(
            edges_a,
            gray_small,
            min_area=min_a,
            white_area=white_a,
            real_aspect_ratio=REAL_ASPECT_RATIO,
            tolerance=ASPECT_TOLERANCE,
            reject_status=reject_a,
        )

        # ---- 自动学习：矩形 ROI / 全图 fallback（每帧学习） ----
        if learn_cooldown > 0:
            learn_cooldown -= 1

        if rect_a is not None and learn_cooldown == 0:
            # 有矩形 → 跟踪稳定性 → ROI 学习
            if last_learned_rect is not None:
                cx_new = rect_a[:, 0].mean()
                cy_new = rect_a[:, 1].mean()
                cx_old = last_learned_rect[:, 0].mean()
                cy_old = last_learned_rect[:, 1].mean()
                dist = np.sqrt((cx_new - cx_old) ** 2 + (cy_new - cy_old) ** 2)
                if dist < 40:
                    stable_count += 1
                else:
                    stable_count = 0
            else:
                stable_count += 1

            last_learned_rect = rect_a.copy()

            if stable_count >= LEARN_STABILITY:
                rect_scaled = (rect_a / scale).astype(np.float32)
                x, y, w, h = cv2.boundingRect(rect_scaled.astype(np.int32))
                # 向外扩展（基于矩形短边的 15%），纳入边框及周围环境
                border = max(8, int(min(w, h) * 0.15))
                fh, fw = frame.shape[:2]
                _do_learn(
                    frame,
                    max(0, x - border),
                    max(0, y - border),
                    min(fw, x + w + border) - max(0, x - border),
                    min(fh, y + h + border) - max(0, y - border),
                    "rect+border",
                )
                learn_cooldown = LEARN_COOLDOWN
                stable_count = 0
                last_learned_rect = None

        elif rect_a is None and learn_cooldown == 0:
            # 无矩形 → 全图中心区域学习（每帧触发，靠 cooldown 限速）
            fh, fw = frame.shape[:2]
            cw, ch = int(fw * 0.4), int(fh * 0.4)
            cx, cy = (fw - cw) // 2, (fh - ch) // 2
            _do_learn(frame, cx, cy, cw, ch, "全图中心")
            learn_cooldown = LEARN_COOLDOWN

        # ---- 管线 B: range + RectTracker ----
        mask_r, edges_r = pipeline_range(frame_small, range_bin, c_lo, c_hi, kernel)
        reject_r: dict = {"area": 0, "quad": 0, "white_region": 0, "aspect_ratio": 0}
        if range_bin.is_learned:
            rect_r = tracker_r.track(
                edges_r,
                gray_small,
                min_area=min_a,
                white_area=white_a,
                real_aspect_ratio=REAL_ASPECT_RATIO,
                tolerance=ASPECT_TOLERANCE,
                reject_status=reject_r,
            )
        else:
            rect_r = None

        # ---- 合成画面 ----
        hdr_h = 40
        canvas = np.zeros((cell_h * 2 + hdr_h, cell_w * 4, 3), dtype=np.uint8)

        # 行1: adaptive
        y0 = 0
        canvas[y0 : y0 + cell_h, :cell_w] = draw_rect(frame_small, rect_a)
        cv2.putText(
            canvas,
            "Adaptive Detect",
            (3, cell_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
        )

        canvas[y0 : y0 + cell_h, cell_w : cell_w * 2] = cv2.cvtColor(
            mask_a, cv2.COLOR_GRAY2BGR
        )
        cv2.putText(
            canvas,
            "Adaptive Mask",
            (cell_w + 3, cell_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
        )

        canvas[y0 : y0 + cell_h, cell_w * 2 : cell_w * 3] = cv2.cvtColor(
            edges_a, cv2.COLOR_GRAY2BGR
        )
        rej_a_text = (
            f"A:{reject_a['area']} Q:{reject_a['quad']} "
            f"W:{reject_a['white_region']} R:{reject_a['aspect_ratio']}"
        )
        cv2.putText(
            canvas,
            f"Edges  {rej_a_text}",
            (cell_w * 2 + 3, cell_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 255, 0),
            1,
        )

        canvas[y0 : y0 + cell_h, cell_w * 3 : cell_w * 4] = draw_rect(
            frame_small, rect_a, (255, 200, 0)
        )
        status_a = "FOUND" if rect_a is not None else "NONE"
        tracking_a = (
            f"hits={tracker_a.stats['track_hits']}  miss={tracker_a.stats['misses']}"
        )
        cv2.putText(
            canvas,
            f"Adaptive: {status_a}  {tracking_a}",
            (cell_w * 3 + 3, cell_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 255, 0) if rect_a is not None else (0, 0, 255),
            1,
        )

        # 行2: range
        y1 = cell_h
        canvas[y1 : y1 + cell_h, :cell_w] = draw_rect(frame_small, rect_r)
        cv2.putText(
            canvas,
            "Range Detect",
            (3, y1 + cell_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
        )

        canvas[y1 : y1 + cell_h, cell_w : cell_w * 2] = cv2.cvtColor(
            mask_r, cv2.COLOR_GRAY2BGR
        )
        cv2.putText(
            canvas,
            "Range Mask",
            (cell_w + 3, y1 + cell_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
        )

        canvas[y1 : y1 + cell_h, cell_w * 2 : cell_w * 3] = cv2.cvtColor(
            edges_r, cv2.COLOR_GRAY2BGR
        )
        rej_r_text = (
            f"A:{reject_r['area']} Q:{reject_r['quad']} "
            f"W:{reject_r['white_region']} R:{reject_r['aspect_ratio']}"
        )
        cv2.putText(
            canvas,
            f"Edges  {rej_r_text}",
            (cell_w * 2 + 3, y1 + cell_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 255, 0),
            1,
        )

        canvas[y1 : y1 + cell_h, cell_w * 3 : cell_w * 4] = draw_rect(
            frame_small, rect_r, (0, 200, 255)
        )
        status_r = "FOUND" if rect_r is not None else "NONE"
        tracking_r = (
            f"hits={tracker_r.stats['track_hits']}  miss={tracker_r.stats['misses']}"
        )
        cv2.putText(
            canvas,
            f"Range: {status_r}  {tracking_r}",
            (cell_w * 3 + 3, y1 + cell_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 255, 0) if rect_r is not None else (0, 0, 255),
            1,
        )

        # 底部信息栏
        y_bar = cell_h * 2 + 6
        if range_bin.is_learned:
            lo, up = range_bin._lower, range_bin._upper
            q = range_bin.quality
            range_info = (
                f"Range LAB: L[{lo[0]},{up[0]}] "
                f"A[{lo[1]},{up[1]}] B[{lo[2]},{up[2]}]  "
                f"Q:{q.get('span_L',0):.0f}/{q.get('span_AB',0):.0f}"
                f"{'⚠纯色' if q.get('is_informative',1)==0 else ''}"
            )
        else:
            range_info = (
                f"Range: 等待学习... rect={stable_count}/{LEARN_STABILITY}"
            )

        cv2.putText(
            canvas,
            f"Canny={c_lo}/{c_hi}  min_area={min_a}  white={white_a}  |  {range_info}",
            (5, y_bar),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            canvas,
            "q/ESC=退出  r=重置  s=截图  a=手动学习",
            (5, y_bar + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (150, 150, 150),
            1,
        )

        cv2.imshow(WIN_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            range_bin._lower = None
            range_bin._upper = None
            tracker_a.reset()
            tracker_r.reset()
            learn_cooldown = 0
            stable_count = 0
            last_learned_rect = None
            cv2.setTrackbarPos("canny_lo", WIN_NAME, 50)
            cv2.setTrackbarPos("canny_hi", WIN_NAME, 150)
            cv2.setTrackbarPos("min_area", WIN_NAME, 20)
            cv2.setTrackbarPos("white_area", WIN_NAME, 10)
            
            print("已重置")
        elif key == ord("a"):
            if rect_a is not None:
                rect_scaled = (rect_a / scale).astype(np.float32)
                try:
                    range_bin.learn_from_rect_pts(frame, rect_scaled, margin=3)
                    print(f"[手动学习] {range_bin.describe()}")
                    tracker_r.reset()
                except RuntimeError as e:
                    print(f"[手动学习失败] {e}")
            else:
                print("adaptive 未检测到矩形，无法学习")
        elif key == ord("s"):
            stamp = cv2.getTickCount()
            path = f"rect_detect_autolearn_{int(stamp)}.png"
            cv2.imwrite(path, canvas)
            print(f"截图: {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
