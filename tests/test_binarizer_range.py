"""Binarizer range 学习模式 — 摄像头实时测试。

鼠标拖选 ROI → 自动学习 LAB 范围 → 实时二值化。

用法::

    python tests/test_binarizer_range.py

操作:
    鼠标拖选      在原图区域框选 ROI，松手后自动学习
    r             清除已学范围
    s             截图保存
    q / ESC       退出
    +/-           调整 min_range

画面: 左上=原图+ROI框  右上=二值化结果  底部=LAB范围值
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

WIN_NAME = "Binarizer | Range"


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
    print("操作: 鼠标拖选 ROI → 自动学习 → 实时二值化")
    print("按键: q/ESC=退出  r=重置  s=截图  +/-=调min_range")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 540)
    cv2.imshow(WIN_NAME, np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.waitKey(100)

    selector = ROISelector()
    cv2.setMouseCallback(WIN_NAME, selector.callback)

    binarizer = Binarizer(strategy="range", min_range=8)

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

        # ROI 学习
        if selector.confirmed is not None and not selector.dragging:
            rx, ry, rw, rh = selector.confirmed
            rx_o, ry_o = int(rx / scale), int(ry / scale)
            rw_o, rh_o = int(rw / scale), int(rh / scale)
            try:
                binarizer.learn(frame, roi_x=rx_o, roi_y=ry_o, roi_w=rw_o, roi_h=rh_o)
                print(f"已学习 ROI ({rx_o},{ry_o}) {rw_o}×{rh_o} — {binarizer.describe()}")
            except RuntimeError as e:
                print(f"学习失败: {e}")
            selector.reset()

        # 二值化
        if binarizer.is_learned:
            mask = binarizer.apply(frame_small)
        else:
            mask = np.zeros((display_h, display_w), dtype=np.uint8)

        # 合成
        hdr_h = 64
        canvas = np.zeros((display_h + hdr_h, display_w * 2, 3), dtype=np.uint8)

        # 左: 原图 + ROI 框
        canvas[:display_h, :display_w] = frame_small
        if selector.confirmed is not None:
            rx, ry, rw, rh = selector.confirmed
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        if selector.dragging:
            cv2.rectangle(canvas, selector.start, selector.end, (0, 255, 255), 1)
        cv2.putText(canvas, "拖选 ROI", (10, display_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 右: 二值化结果
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        canvas[:display_h, display_w:display_w * 2] = mask_bgr
        cv2.putText(canvas, "Binary", (display_w + 10, display_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 底部信息
        y0 = display_h + 20
        if binarizer.is_learned:
            lo, up = binarizer._lower, binarizer._upper
            cv2.putText(canvas,
                        f"min_range={binarizer.min_range}  |  "
                        f"L=[{lo[0]},{up[0]}]  A=[{lo[1]},{up[1]}]  B=[{lo[2]},{up[2]}]",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        else:
            cv2.putText(canvas, "未学习 — 鼠标拖选 ROI", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(canvas, "q/ESC=退出  r=重置  s=截图  +/-=min_range",
                    (10, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow(WIN_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            binarizer._lower = None
            binarizer._upper = None
            selector.reset()
            print("已重置")
        elif key == ord("s"):
            stamp = cv2.getTickCount()
            path = f"binarizer_range_{int(stamp)}.png"
            cv2.imwrite(path, canvas)
            print(f"截图: {path}")
        elif key in (ord("+"), ord("=")):
            binarizer.min_range = min(binarizer.min_range + 2, 64)
            print(f"min_range = {binarizer.min_range}")
        elif key == ord("-"):
            binarizer.min_range = max(binarizer.min_range - 2, 2)
            print(f"min_range = {binarizer.min_range}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
