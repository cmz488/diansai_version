"""Binarizer 策略对比 — 摄像头实时 adaptive vs range。

左侧 adaptive 实时二值化，右侧用鼠标选 ROI 学习 range 后对比。

用法::

    python tests/test_binarizer_compare.py

操作:
    鼠标拖选 ROI     → range 学习（右侧更新）
    Trackbar 调参     → adaptive 调参（左侧更新）
    q / ESC          退出
    r                重置
    s                截图

画面: 原图(上)  左=adaptive红蒙版  右=range红蒙版  底部=参数对比
"""

import os
import sys
from pathlib import Path

# 强制使用 X11 后端，避免 Wayland/Qt 字体冲突
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ.pop("WAYLAND_DISPLAY", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from tools._threshold import Binarizer

# ============================================================================

WIN_NAME = "Binarizer | Compare"


def nothing(_: int) -> None:
    pass


class ROISelector:
    def __init__(self) -> None:
        self.dragging = False
        self.start = (0, 0)
        self.end = (0, 0)
        self.confirmed: tuple[int, int, int, int] | None = None

    def reset(self) -> None:
        self.dragging = False
        self.confirmed = None

    def callback(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.end = (x, y)
            x1, y1 = self.start
            x2, y2 = self.end
            x0, y0 = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w >= 10 and h >= 10:
                self.confirmed = (x0, y0, w, h)


def overlay_red(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """在 BGR 图上叠加红色半透明掩码。"""
    if not np.any(mask):
        return img.copy()
    out = img.copy()
    red = np.zeros_like(img)
    red[:, :, 2] = 255
    fg = mask > 0
    # OpenCV 5 对 boolean-indexed 数组的 addWeighted 行为变化，
    # 改为全图混合后取前景像素
    blended = cv2.addWeighted(img, 1 - alpha, red, alpha, 0)
    out[fg] = blended[fg]
    return out


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头 (device 0)")
        sys.exit(1)

    ok, frame = cap.read()
    if not ok:
        print("无法读取摄像头帧")
        sys.exit(1)

    print(f"摄像头已连接  {frame.shape[1]}×{frame.shape[0]}")
    print("Trackbar: morph_k=形态学核  gauss_k=高斯核  min_range=范围学习容差")
    print("操作: 鼠标拖选 ROI → range 学习")
    print("按键: q/ESC=退出  r=重置  s=截图")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 800)
    cv2.imshow(WIN_NAME, np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.waitKey(100)

    selector = ROISelector()
    cv2.setMouseCallback(WIN_NAME, selector.callback)

    cv2.createTrackbar("morph_k", WIN_NAME, 5, 21, nothing)
    cv2.createTrackbar("gauss_k", WIN_NAME, 5, 21, nothing)
    cv2.createTrackbar("min_range", WIN_NAME, 8, 64, nothing)

    adaptive = Binarizer(strategy="adaptive")
    range_bin = Binarizer(strategy="range", min_range=8)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display_w = 640
        scale = display_w / w
        display_h = int(h * scale)
        frame_small = cv2.resize(frame, (display_w, display_h))

        # trackbar
        mk = cv2.getTrackbarPos("morph_k", WIN_NAME)
        gk = cv2.getTrackbarPos("gauss_k", WIN_NAME)
        mr = cv2.getTrackbarPos("min_range", WIN_NAME)
        mk = max(3, mk + 1 if mk % 2 == 0 else mk)
        gk = max(3, gk + 1 if gk % 2 == 0 else gk)
        mr = max(2, mr)

        adaptive.morph_kernel = (mk, mk)
        adaptive.gauss_kernel = (gk, gk)
        range_bin.min_range = mr

        # ROI 学习
        if selector.confirmed is not None and not selector.dragging:
            rx, ry, rw, rh = selector.confirmed
            rx_o, ry_o = int(rx / scale), int(ry / scale)
            rw_o, rh_o = int(rw / scale), int(rh / scale)
            try:
                range_bin.learn(frame, roi_x=rx_o, roi_y=ry_o, roi_w=rw_o, roi_h=rh_o)
                print(f"range 已学习: {range_bin.describe()}")
            except RuntimeError as e:
                print(f"学习失败: {e}")
            selector.reset()

        # 二值化
        mask_a = adaptive.apply(frame_small)
        mask_r = range_bin.apply(frame_small) if range_bin.is_learned else \
                 np.zeros_like(mask_a)

        a_info = adaptive.params

        # 统计
        a_white = np.sum(mask_a > 0) / mask_a.size * 100
        r_white = np.sum(mask_r > 0) / mask_r.size * 100
        overlap = np.sum((mask_a > 0) & (mask_r > 0))
        union = np.sum((mask_a > 0) | (mask_r > 0))
        iou = overlap / union * 100 if union > 0 else 0

        # 合成
        hdr_h = 80
        src_h = display_h // 2
        bot_h = display_h - src_h
        canvas = np.zeros((display_h + hdr_h, display_w * 2, 3), dtype=np.uint8)

        # 顶部: 原图
        canvas[:src_h, :display_w] = frame_small[:src_h]
        canvas[:src_h, display_w:] = frame_small[:src_h]
        if selector.confirmed is not None:
            rx, ry, rw, rh = selector.confirmed
            if ry < src_h:
                cv2.rectangle(canvas, (rx, ry),
                              (min(rx + rw, display_w), min(ry + rh, src_h)),
                              (0, 255, 0), 2)
        if selector.dragging:
            cv2.rectangle(canvas, selector.start,
                          (min(selector.end[0], display_w), min(selector.end[1], src_h)),
                          (0, 255, 255), 1)
        cv2.putText(canvas, "拖选 ROI → range 学习", (10, src_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 左下: adaptive overlay
        y_bot = src_h
        roi_a = frame_small[src_h:src_h + bot_h]
        roi_m_a = mask_a[src_h:src_h + bot_h]
        canvas[y_bot:y_bot + bot_h, :display_w] = overlay_red(roi_a, roi_m_a)
        cv2.putText(canvas,
                    f"Adaptive  C={a_info['C']}  bs={a_info['block_size']}  "
                    f"白色={a_white:.1f}%",
                    (5, y_bot + bot_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 2)

        # 右下: range overlay
        roi_r = frame_small[src_h:src_h + bot_h]
        roi_m_r = mask_r[src_h:src_h + bot_h]
        canvas[y_bot:y_bot + bot_h, display_w:] = overlay_red(roi_r, roi_m_r)
        if range_bin.is_learned:
            lo, up = range_bin._lower, range_bin._upper
            desc = f"L[{lo[0]},{up[0]}] A[{lo[1]},{up[1]}] B[{lo[2]},{up[2]}]"
        else:
            desc = "未学习"
        cv2.putText(canvas,
                    f"Range  {desc}  白色={r_white:.1f}%  IoU={iou:.1f}%",
                    (display_w + 5, y_bot + bot_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 2)

        # 底部信息条
        y_bar = display_h + 20
        cv2.putText(canvas,
                    f"morph={mk}  gauss={gk}  min_range={mr}  |  q/ESC=退出  r=重置  s=截图",
                    (10, y_bar), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow(WIN_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            cv2.setTrackbarPos("morph_k", WIN_NAME, 5)
            cv2.setTrackbarPos("gauss_k", WIN_NAME, 5)
            cv2.setTrackbarPos("min_range", WIN_NAME, 8)
            range_bin._lower = None
            range_bin._upper = None
            selector.reset()
            print("已重置")
        elif key == ord("s"):
            stamp = cv2.getTickCount()
            path = f"binarizer_compare_{int(stamp)}.png"
            cv2.imwrite(path, canvas)
            print(f"截图: {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
