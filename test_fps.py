"""Measure real OpenCV video-stream FPS with warmup and latency statistics."""

from __future__ import annotations

import argparse
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.frames <= 0 or args.warmup < 0:
        raise ValueError("frames 必须 > 0，warmup 必须 >= 0")

    capture = JetsonCamera(
        PipelineConfig(
            source=args.source,
            device=args.device,
            sensor_id=args.sensor_id,
            sensor_mode=args.sensor_mode,
            width=args.width,
            height=args.height,
            fps=args.fps,
            flip_method=args.flip_method,
        ),
    )
    if not capture.open():
        raise RuntimeError(f"无法打开视频流：{capture.last_error}")

    try:
        for _ in range(args.warmup):
            ok, _ = capture.read()
            if not ok:
                raise RuntimeError("预热阶段读取失败")

        latencies = []
        valid_frames = 0
        failed_reads = 0
        started = time.perf_counter()
        while valid_frames < args.frames:
            frame_started = time.perf_counter()
            ok, frame = capture.read()
            if not ok or frame is None:
                failed_reads += 1
                if failed_reads >= 30:
                    raise RuntimeError("连续读取失败")
                continue
            failed_reads = 0
            valid_frames += 1
            latencies.append(time.perf_counter() - frame_started)
        elapsed = time.perf_counter() - started
    finally:
        capture.release()

    ordered = sorted(latencies)
    p95 = ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
    print(f"pipeline: {'argus_vic' if args.source == 'csi' else 'nvdec_vic'}")
    print(f"frames: {valid_frames}")
    print(f"elapsed: {elapsed:.3f} s")
    print(f"fps: {valid_frames / elapsed:.3f}")
    print(f"read mean: {statistics.mean(latencies) * 1000:.3f} ms")
    print(f"read p50: {statistics.median(latencies) * 1000:.3f} ms")
    print(f"read p95: {p95 * 1000:.3f} ms")


if __name__ == "__main__":
    main()
