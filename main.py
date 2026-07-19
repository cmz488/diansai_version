"""Hardware-accelerated camera capture and OpenCV target detection."""

from __future__ import annotations

import argparse
import time
from typing import Optional, Sequence

import cv2
import numpy as np

from tools.hardware_pipeline import JetsonCamera, PipelineConfig
from tools.tools import (
    FpsShow,
    cvt_mvlab2cv,
    detect_rect,
    preprocess,
)
from tools.web import DebugServer, ParamRegistry


REAL_ASPECT_RATIO = 0.657
ASPECT_TOLERANCE = 0.4


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("csi", "usb"), default="csi")
    parser.add_argument("--device", default="/dev/video0", help="USB 模式的视频节点")
    parser.add_argument("--sensor-id", type=int, default=0, help="CSI/Argus 传感器编号")
    parser.add_argument("--sensor-mode", type=int, default=-1, help="-1 表示自动选择")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--flip-method", type=int, default=6, choices=range(8))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0 表示持续运行；正数用于测试后自动退出",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = PipelineConfig(
        source=args.source,
        device=args.device,
        sensor_id=args.sensor_id,
        sensor_mode=args.sensor_mode,
        width=args.width,
        height=args.height,
        fps=args.fps,
        flip_method=args.flip_method,
    )
    camera = JetsonCamera(config)
    if not camera.open():
        raise RuntimeError(f"无法打开视频流：{camera.last_error}")

    print(
        f"[pipeline] {config.source} -> VIC -> OpenCV BGR, "
        f"size={config.width}x{config.height}@{config.fps}, "
        f"sensor-id={config.sensor_id}" if config.source == "csi" else
        f"[pipeline] usb -> NVDEC/VIC -> OpenCV BGR, device={config.device}, "
        f"size={config.width}x{config.height}@{config.fps}"
    )

    params = ParamRegistry()
    params.add("kernel", type=int, default=5, range=(1, 21), step=2, group="形态学")
    params.add("min_area", type=int, default=2000, range=(100, 20000), group="筛选")
    params.add("min_white", type=int, default=60, range=(0, 255), group="筛选")

    server: Optional[DebugServer] = None
    if not args.no_web:
        server = DebugServer(params=params, port=args.port)
        server.start()

    fps = FpsShow()
    rect_lab_lower, rect_lab_upper = cvt_mvlab2cv([41, 74, -14, 13, -27, 31])

    kernel_size = -1
    kernel = np.ones((5, 5), dtype=np.uint8)
    consecutive_read_failures = 0
    processed_frames = 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures >= 30:
                    raise RuntimeError("连续 30 帧读取失败，视频流可能已断开")
                time.sleep(0.005)
                continue
            consecutive_read_failures = 0
            raw_frame = frame.copy()
            requested_kernel = int(params.get("kernel"))
            if requested_kernel != kernel_size:
                if requested_kernel % 2 == 0:
                    requested_kernel += 1
                kernel_size = requested_kernel
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (kernel_size, kernel_size),
                )

            min_area = int(params.get("min_area"))
            min_white = int(params.get("min_white"))
            rect_edges, gray = preprocess(
                frame,
                kernel,
                (rect_lab_lower, rect_lab_upper),
            )

            reject_status = {
                "area": 0,
                "quad": 0,
                "white_region": 0,
                "aspect_ratio": 0,
            }
            best_rect = detect_rect(
                rect_edges,
                gray,
                min_area,
                min_white,
                REAL_ASPECT_RATIO,
                tolerance=ASPECT_TOLERANCE,
                reject_status=reject_status,
            )


            rect_center_x: Optional[int] = None
            rect_center_y: Optional[int] = None
            if best_rect is not None:
                rect_center_x = int(best_rect[:, 0].mean())
                rect_center_y = int(best_rect[:, 1].mean())
                cv2.polylines(frame, [best_rect], True, (0, 255, 0), 2)
                cv2.circle(
                    frame,
                    (rect_center_x, rect_center_y),
                    5,
                    (0, 255, 0),
                    -1,
                )

            if rect_center_x is not None and rect_center_y is not None:
                cv2.putText(
                    frame,
                    f"rect:({rect_center_x},{rect_center_y})",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            fps.show(frame)
            cv2.putText(
                frame,
                f"area:{reject_status['area']} "
                f"asp:{reject_status['aspect_ratio']} "
                f"quad:{reject_status['quad']} "
                f"white:{reject_status['white_region']}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            if server is not None:
                server.metrics.update(
                    fps=round(fps.fps, 2),
                    capture_path="argus_vic" if config.source == "csi" else "nvdec_vic",
                )
                server.broadcast_metrics()
                server.update_frame(0, frame)
                server.update_frame(1, rect_edges)

            processed_frames += 1
            if args.max_frames > 0 and processed_frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\n[main] 收到退出信号")
    finally:
        camera.release()
        if server is not None:
            server.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
