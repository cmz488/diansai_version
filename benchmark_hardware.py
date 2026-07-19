"""Benchmark the selected Jetson CSI or USB hardware capture path."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Optional, Sequence

from tools.hardware_pipeline import JetsonCamera, PipelineConfig


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("csi", "usb"), default="csi")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--sensor-mode", type=int, default=-1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--flip-method", type=int, default=6, choices=range(8))
    parser.add_argument("--json", type=str)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.frames <= 0 or args.warmup < 0:
        raise ValueError("frames 必须 > 0，warmup 必须 >= 0")

    camera = JetsonCamera(
        PipelineConfig(
            source=args.source,
            device=args.device,
            sensor_id=args.sensor_id,
            sensor_mode=args.sensor_mode,
            width=args.width,
            height=args.height,
            fps=args.fps,
            flip_method=args.flip_method,
        )
    )
    if not camera.open():
        raise RuntimeError(f"无法打开视频流：{camera.last_error}")

    try:
        for _ in range(args.warmup):
            ok, _ = camera.read()
            if not ok:
                raise RuntimeError("预热阶段读取失败")

        latencies = []
        started = time.perf_counter()
        for _ in range(args.frames):
            frame_started = time.perf_counter()
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("读取视频帧失败")
            latencies.append(time.perf_counter() - frame_started)
        elapsed = time.perf_counter() - started
    finally:
        camera.release()

    ordered = sorted(latencies)
    result = {
        "pipeline": "argus_vic" if args.source == "csi" else "nvdec_vic",
        "frames": args.frames,
        "elapsed_s": round(elapsed, 6),
        "fps": round(args.frames / elapsed, 3),
        "read_mean_ms": round(statistics.mean(latencies) * 1000, 3),
        "read_p95_ms": round(ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)] * 1000, 3),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
