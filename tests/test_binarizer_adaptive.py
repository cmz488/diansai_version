"""Binarizer adaptive 模式 — 摄像头实时测试。

用法::

    python tests/test_binarizer_adaptive.py

按键:
    q / ESC   退出
    1 / 2     切换步长
    r         重置参数
    s         截图保存
    m         切换镜像

Trackbar:
    morph_k   形态学核大小 (3~21, 奇数)
    gauss_k   高斯核大小 (3~21, 奇数)

画面: 左=原图  右=二值化结果  底部=实时参数
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

WIN_NAME = "Binarizer | Adaptive"


def nothing(_: int) -> None:
    pass


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头 (device 0)")
        sys.exit(1)

    # 读取一帧获取尺寸
    ok, frame = cap.read()
    if not ok:
        print("无法读取摄像头帧")
        sys.exit(1)

    print(f"摄像头已连接  {frame.shape[1]}×{frame.shape[0]}")
    print("Trackbar: morph_k=形态学核  gauss_k=高斯核")
    print("按键: q/ESC=退出  r=重置  s=截图  m=镜像")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 540)
    # 显式显示一帧确保窗口后端完全初始化
    cv2.imshow(WIN_NAME, np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.waitKey(100)

    cv2.createTrackbar("morph_k", WIN_NAME, 5, 21, nothing)
    cv2.createTrackbar("gauss_k", WIN_NAME, 5, 21, nothing)

    binarizer = Binarizer(strategy="adaptive")
    mirror = True

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        display_w = 640
        scale = display_w / w
        display_h = int(h * scale)
        frame_small = cv2.resize(frame, (display_w, display_h))

        # trackbar
        mk = cv2.getTrackbarPos("morph_k", WIN_NAME)
        gk = cv2.getTrackbarPos("gauss_k", WIN_NAME)
        mk = max(3, mk + 1 if mk % 2 == 0 else mk)
        gk = max(3, gk + 1 if gk % 2 == 0 else gk)

        binarizer.morph_kernel = (mk, mk)
        binarizer.gauss_kernel = (gk, gk)

        # 二值化
        mask = binarizer.apply(frame_small)

        # 合成
        info = binarizer.params
        hdr_h = 64
        canvas = np.zeros((display_h + hdr_h, display_w * 2, 3), dtype=np.uint8)

        canvas[:display_h, :display_w] = frame_small
        cv2.putText(canvas, "Original", (10, display_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        canvas[:display_h, display_w:display_w * 2] = mask_bgr
        cv2.putText(canvas, "Binary", (display_w + 10, display_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 信息栏
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        C_val = info.get("C", "?")
        bs_val = info.get("block_size", "?")

        y0 = display_h + 20
        cv2.putText(canvas,
                    f"brightness={brightness:.0f}  |  C={C_val}  block_size={bs_val}  "
                    f"|  morph={mk}  gauss={gk}",
                    (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(canvas,
                    "q/ESC=退出  r=重置  s=截图  m=镜像",
                    (10, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow(WIN_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            cv2.setTrackbarPos("morph_k", WIN_NAME, 5)
            cv2.setTrackbarPos("gauss_k", WIN_NAME, 5)
        elif key == ord("s"):
            stamp = cv2.getTickCount()
            path = f"binarizer_adaptive_{int(stamp)}.png"
            cv2.imwrite(path, canvas)
            print(f"截图: {path}")
        elif key == ord("m"):
            mirror = not mirror
            print(f"镜像: {'ON' if mirror else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
