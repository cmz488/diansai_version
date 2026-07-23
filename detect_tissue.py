"""纸巾包装检测 — 用 CamShift 颜色直方图追踪指定目标。

用法:
    python detect_tissue.py [CAMERA_INDEX]

操作:
    1. 鼠标框选目标区域（按住左键拖拽）
    2. 按 SPACE 开始追踪
    3. 按 r 重新选区域
    4. 按 q 退出
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

from tools._camshift import CamShiftTracker

# ============================================================================
# 参数
# ============================================================================

CAMERA_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# ============================================================================
# 鼠标框选
# ============================================================================


class ROISelector:
    """用鼠标在画面上框选矩形 ROI。"""

    def __init__(self, window_name: str):
        self.window = window_name
        self._drawing = False
        self._start = (0, 0)
        self._end = (0, 0)
        self.bbox: tuple | None = None
        cv2.setMouseCallback(window_name, self._on_mouse)

    def _on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start = (x, y)
            self._end = (x, y)
            self.bbox = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            self._end = (x, y)
            x1, y1 = self._start
            x2, y2 = self._end
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            if w > 10 and h > 10:
                self.bbox = (x, y, w, h)

    def draw(self, frame: np.ndarray):
        if self._drawing:
            cv2.rectangle(frame, self._start, self._end, (0, 255, 0), 2)
        elif self.bbox is not None:
            x, y, w, h = self.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


# ============================================================================
# 主程序
# ============================================================================


def main():
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(f"无法打开相机 /dev/video{CAMERA_INDEX}")

    win = "Tissue Tracker"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    selector = ROISelector(win)
    tracker = CamShiftTracker(margin=0.2, max_misses=15)
    tracking = False

    print("操作: 鼠标框选目标 → SPACE 开始追踪 → r 重选 → q 退出")

    while True:
        ok, frame = camera.read()
        if not ok:
            print("[warn] 相机读取失败，重试中...")
            continue

        if tracking and tracker.ready:
            # ── 追踪模式 ──
            bbox = tracker.predict(frame)
            if bbox is not None:
                x, y, w, h = bbox
                # 画追踪框（红色）
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # 画 CamShift 中心十字
                cx = x + w // 2
                cy = y + h // 2
                cv2.drawMarker(
                    frame, (cx, cy), (0, 0, 255),
                    cv2.MARKER_CROSS, 20, 2,
                )
            else:
                # 丢失
                cv2.putText(
                    frame, "LOST", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                )
                if not tracker.ready:
                    tracking = False
        else:
            # ── 选区域模式 ──
            selector.draw(frame)
            cv2.putText(
                frame, "SELECT ROI then SPACE", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

        # 状态栏
        status = f"tracking={tracking} ready={tracker.ready} misses={tracker.miss_count}"
        cv2.putText(
            frame, status, (50, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" ") and selector.bbox is not None:
            # 开始追踪
            tracker.init(frame, selector.bbox)
            tracking = True
            print(f"[init] ROI={selector.bbox}")
        elif key == ord("r"):
            # 重置
            tracking = False
            tracker.reset()
            selector.bbox = None
            print("[reset] 重新选择目标")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
