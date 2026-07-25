#!/usr/bin/env python3
"""使用 YoloDetector + OpenCV 对视频流进行 ball 实时识别。

示例::

    conda activate yolo
    python tests/test_yolo_stream.py --source 0
    python tests/test_yolo_stream.py --source video.mp4
    python tests/test_yolo_stream.py --source rtsp://user:pass@host/stream
    python tests/test_yolo_stream.py --source 0 --output runs/ball_stream.mp4

按键：q / Esc 退出，空格暂停或继续。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.yolo_detector import BBox, YoloDetector


BALL_CLASS_ID = 0
BALL_LABEL = "ball"
WINDOW_NAME = "YoloDetector | Ball Stream"


def find_latest_best() -> Path:
    """自动选择最近生成的训练最佳权重。"""
    candidates = list((PROJECT_ROOT / "runs" / "detect").glob("**/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            "找不到 runs/detect/**/weights/best.pt，请先训练或使用 --weights 指定权重"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def project_path(value: str) -> Path:
    """将相对文件路径按项目根目录解析。"""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_source(value: str) -> int | str:
    """纯数字源按摄像头编号处理，其他值按文件或网络流处理。"""
    return int(value) if value.isdigit() else value


def draw_ball_boxes(
    frame: np.ndarray,
    boxes: list[BBox],
    class_id: int = BALL_CLASS_ID,
) -> int:
    """在帧上绘制 ball 检测框，返回绘制数量。"""
    ball_boxes = [box for box in boxes if box.cls_id == class_id]
    for box in ball_boxes:
        color = (0, 255, 0)
        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), color, 2)

        center = ((box.x1 + box.x2) // 2, (box.y1 + box.y2) // 2)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        text = f"{BALL_LABEL} {box.conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        text_y = max(box.y1, text_h + baseline + 4)
        cv2.rectangle(
            frame,
            (box.x1, text_y - text_h - baseline - 4),
            (box.x1 + text_w + 6, text_y),
            color,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (box.x1 + 3, text_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return len(ball_boxes)


def draw_status(frame: np.ndarray, fps: float, count: int, backend: str) -> None:
    """绘制实时状态。"""
    text = f"FPS:{fps:.1f}  balls:{count}  backend:{backend}"
    cv2.putText(
        frame,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 tools.yolo_detector 识别视频流中的 ball"
    )
    parser.add_argument(
        "--source",
        default="3",
        help="摄像头编号、视频文件或 RTSP/HTTP 地址（默认: 0）",
    )
    parser.add_argument(
        "--weights",
        help="PyTorch best.pt；默认自动选择 runs/detect 下最新权重",
    )
    parser.add_argument(
        "--engine",
        help="可选 TensorRT .engine；加载失败时回退到 --weights",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理尺寸")
    parser.add_argument(
        "--class-id", type=int, default=BALL_CLASS_ID, help="ball 类别 ID"
    )
    parser.add_argument("--width", type=int, help="摄像头采集宽度")
    parser.add_argument("--height", type=int, help="摄像头采集高度")
    parser.add_argument("--fps", type=float, help="摄像头目标帧率")
    parser.add_argument("--output", help="可选：保存带标注的输出视频")
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="使用 OpenCV 窗口实时显示",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="最多处理帧数，0 表示无限制",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = parse_source(args.source)
    weights = project_path(args.weights) if args.weights else find_latest_best()
    engine = project_path(args.engine) if args.engine else None

    if not weights.is_file():
        raise FileNotFoundError(f"找不到 PyTorch 权重: {weights}")
    if engine is not None and not engine.is_file():
        print(f"[warn] TensorRT engine 不存在，将回退到 PyTorch: {engine}")

    detector = YoloDetector(
        engine_path=str(engine) if engine is not None else None,
        pt_path=str(weights),
        conf=args.conf,
        imgsz=args.imgsz,
    )

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频流: {args.source}")

    if isinstance(source, int):
        if args.width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.fps:
            capture.set(cv2.CAP_PROP_FPS, args.fps)

    writer: cv2.VideoWriter | None = None
    output_path = project_path(args.output) if args.output else None
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0 or source_fps > 240:
        source_fps = args.fps or 30.0

    print(f"[stream] source:  {args.source}")
    print(f"[stream] weights: {weights}")
    print(f"[stream] class:   {args.class_id} ({BALL_LABEL})")
    print(f"[stream] conf:    {args.conf}")
    print("[stream] q/Esc=退出，Space=暂停/继续")

    if args.display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_count = 0
    smoothed_fps = 0.0
    previous_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            boxes = detector.detect(frame)
            ball_count = draw_ball_boxes(frame, boxes, args.class_id)

            current_time = time.perf_counter()
            elapsed = max(current_time - previous_time, 1e-9)
            previous_time = current_time
            instant_fps = 1.0 / elapsed
            smoothed_fps = (
                instant_fps
                if smoothed_fps == 0.0
                else 0.9 * smoothed_fps + 0.1 * instant_fps
            )
            draw_status(frame, smoothed_fps, ball_count, detector.backend)

            if output_path is not None and writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建输出视频: {output_path}")
            if writer is not None:
                writer.write(frame)

            if args.display:
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused_key = cv2.waitKey(0) & 0xFF
                    if paused_key in (ord("q"), 27):
                        break

            frame_count += 1
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"[ok] 已处理 {frame_count} 帧")
    if output_path is not None:
        print(f"[ok] 输出视频: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
