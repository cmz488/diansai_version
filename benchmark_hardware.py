"""Benchmark Jetson capture/compute backend combinations on the real project pipeline."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tools.hardware_pipeline import (
    OpenCVHardwarePipeline,
    PipelineConfig,
    probe_capabilities,
)
from tools.tools import cvt_mvlab2cv, detect_laser_mask, detect_rect


REAL_ASPECT_RATIO = 0.657
ASPECT_TOLERANCE = 0.4
LASER_MV_LABVALUE = [31, 58, 11, 63, -19, 20]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--flip-method", type=int, default=6, choices=range(8))
    parser.add_argument("--json", type=Path, default=None, help="可选 JSON 结果路径")
    parser.add_argument("--skip-pycuda", action="store_true")
    parser.add_argument(
        "--single-case",
        nargs=2,
        metavar=("CAPTURE", "COMPUTE"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def percentile(values: List[float], ratio: float) -> float:
    ordered = sorted(values)
    index = min(max(int(len(ordered) * ratio), 0), len(ordered) - 1)
    return ordered[index]


def pycuda_self_test() -> Dict[str, object]:
    """Run PyCUDA in a child process so its context cannot poison VPI."""
    env = os.environ.copy()
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
    env.setdefault("CUDA_ROOT", "/usr/local/cuda")
    code = r"""
import json
import numpy as np
import pycuda
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

module = SourceModule(
    "__global__ void add(const float *a, const float *b, float *c, int n) {"
    "int i = blockIdx.x * blockDim.x + threadIdx.x;"
    "if (i < n) c[i] = a[i] + b[i];"
    "}"
)
function = module.get_function("add")
count = 1 << 20
a = np.arange(count, dtype=np.float32)
b = np.full(count, 2.0, dtype=np.float32)
c = np.empty_like(a)
start = cuda.Event()
end = cuda.Event()
start.record()
function(
    cuda.In(a), cuda.In(b), cuda.Out(c), np.int32(count),
    block=(256, 1, 1), grid=((count + 255) // 256, 1, 1),
)
end.record()
end.synchronize()
print(json.dumps({
    "version": pycuda.VERSION_TEXT,
    "device": cuda.Device(0).name(),
    "compute_capability": cuda.Device(0).compute_capability(),
    "verified": bool(np.allclose(c, a + b)),
    "kernel_ms": round(float(start.time_till(end)), 4),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "PyCUDA child failed")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def run_case(
    config: PipelineConfig,
    capture_backend: str,
    compute_backend: str,
    frame_count: int,
    warmup_count: int,
) -> Dict[str, object]:
    pipeline = OpenCVHardwarePipeline(
        config,
        capture_backend=capture_backend,
        compute_backend=compute_backend,
    )
    if not pipeline.open():
        raise RuntimeError(pipeline.capture.last_error or "open failed")
    active_capture = pipeline.capture_backend

    rect_lower, rect_upper = cvt_mvlab2cv()
    laser_lower, laser_upper = cvt_mvlab2cv(LASER_MV_LABVALUE)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for _ in range(warmup_count):
        ok, _ = pipeline.read()
        if not ok:
            pipeline.release()
            raise RuntimeError("warmup read failed")

    capture_times: List[float] = []
    compute_times: List[float] = []
    frame_times: List[float] = []
    rect_hits = 0
    laser_hits = 0
    failed_reads = 0
    started = time.perf_counter()

    try:
        while len(frame_times) < frame_count:
            frame_started = time.perf_counter()
            read_started = time.perf_counter()
            ok, frame = pipeline.read()
            capture_times.append(time.perf_counter() - read_started)
            if not ok or frame is None:
                failed_reads += 1
                if failed_reads >= 30:
                    raise RuntimeError("连续读取失败")
                continue
            failed_reads = 0

            compute_started = time.perf_counter()
            rect_edges, laser_binary, gray = pipeline.preprocess(
                frame,
                kernel,
                (rect_lower, rect_upper),
                (laser_lower, laser_upper),
            )
            reject_status = {
                "area": 0,
                "quad": 0,
                "white_region": 0,
                "aspect_ratio": 0,
            }
            rect = detect_rect(
                rect_edges,
                gray,
                2000,
                60,
                REAL_ASPECT_RATIO,
                tolerance=ASPECT_TOLERANCE,
                reject_status=reject_status,
            )
            laser = detect_laser_mask(laser_binary, 200)
            if rect is not None:
                rect_hits += 1
                cv2.polylines(frame, [rect], True, (0, 255, 0), 2)
            if laser is not None:
                laser_hits += 1
                x, y, w, h = cv2.boundingRect(laser)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame,
                "benchmark",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            compute_times.append(time.perf_counter() - compute_started)
            frame_times.append(time.perf_counter() - frame_started)
    finally:
        pipeline.release()

    elapsed = time.perf_counter() - started
    return {
        "capture_requested": capture_backend,
        "capture_active": active_capture or capture_backend,
        "compute_requested": compute_backend,
        "compute_active": pipeline.compute_backend,
        "frames": len(frame_times),
        "elapsed_s": round(elapsed, 4),
        "end_to_end_fps": round(len(frame_times) / elapsed, 3),
        "algorithm_fps": round(len(compute_times) / sum(compute_times), 3),
        "capture_mean_ms": round(statistics.mean(capture_times) * 1000, 3),
        "compute_mean_ms": round(statistics.mean(compute_times) * 1000, 3),
        "frame_p50_ms": round(statistics.median(frame_times) * 1000, 3),
        "frame_p95_ms": round(percentile(frame_times, 0.95) * 1000, 3),
        "rect_hits": rect_hits,
        "laser_hits": laser_hits,
    }


def run_case_isolated(
    args: argparse.Namespace,
    capture_backend: str,
    compute_backend: str,
) -> Dict[str, object]:
    """Run one backend combination in a fresh process.

    Jetson multimedia and VPI contexts are not always safe to tear down and
    recreate repeatedly in one Python process. Process isolation makes the
    benchmark deterministic and keeps one backend failure from losing all data.
    """
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-case",
        capture_backend,
        compute_backend,
        "--device",
        args.device,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--fps",
        str(args.fps),
        "--frames",
        str(args.frames),
        "--warmup",
        str(args.warmup),
        "--flip-method",
        str(args.flip_method),
        "--skip-pycuda",
    ]
    last_detail = "unknown error"
    for attempt in range(1, 4):
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        result_line = next(
            (
                line
                for line in reversed(completed.stdout.splitlines())
                if line.startswith("CASE_JSON ")
            ),
            None,
        )
        if completed.returncode == 0 and result_line is not None:
            return json.loads(result_line[len("CASE_JSON ") :])
        detail = completed.stderr.strip() or completed.stdout.strip()
        last_detail = f"attempt={attempt}, exit={completed.returncode}: {detail[-1200:]}"
        if attempt < 3:
            time.sleep(2.0)
    raise RuntimeError(last_detail)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.frames <= 0 or args.warmup < 0:
        raise ValueError("frames 必须 > 0，warmup 必须 >= 0")

    config = PipelineConfig(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        flip_method=args.flip_method,
    )
    if args.single_case is not None:
        capture_backend, compute_backend = args.single_case
        if capture_backend not in ("nvdec_vic", "v4l2"):
            raise ValueError(f"未知 capture backend: {capture_backend}")
        if compute_backend not in ("cpu", "umat", "vpi_cuda"):
            raise ValueError(f"未知 compute backend: {compute_backend}")
        result = run_case(
            config,
            capture_backend,
            compute_backend,
            args.frames,
            args.warmup,
        )
        print("CASE_JSON " + json.dumps(result, ensure_ascii=False), flush=True)
        return

    capabilities = probe_capabilities()
    print("CAPABILITIES")
    print(json.dumps(capabilities.to_dict(), ensure_ascii=False, indent=2))

    pycuda_result: Optional[Dict[str, object]] = None
    if capabilities.pycuda and not args.skip_pycuda:
        try:
            pycuda_result = pycuda_self_test()
            print("PYCUDA", json.dumps(pycuda_result, ensure_ascii=False))
        except Exception as exc:
            print(f"PYCUDA SKIP: {type(exc).__name__}: {exc}")

    cases: List[Tuple[str, str]] = []
    if capabilities.gstreamer and capabilities.nvdec and capabilities.vic:
        cases.append(("nvdec_vic", "cpu"))
        if capabilities.vpi:
            cases.append(("nvdec_vic", "vpi_cuda"))
        if capabilities.opencl_enabled:
            cases.append(("nvdec_vic", "umat"))
    cases.append(("v4l2", "cpu"))
    if not (capabilities.gstreamer and capabilities.nvdec and capabilities.vic):
        if capabilities.opencl_enabled:
            cases.append(("v4l2", "umat"))

    results: List[Dict[str, object]] = []
    for capture_backend, compute_backend in cases:
        name = f"{capture_backend}+{compute_backend}"
        print(f"\nRUN {name}")
        try:
            result = run_case_isolated(
                args,
                capture_backend,
                compute_backend,
            )
        except Exception as exc:
            print(f"SKIP {name}: {type(exc).__name__}: {exc}")
            time.sleep(2.0)
            continue
        results.append(result)
        print(
            f"RESULT {name}: end_to_end={result['end_to_end_fps']} FPS, "
            f"algorithm={result['algorithm_fps']} FPS, "
            f"p95={result['frame_p95_ms']} ms"
        )
        time.sleep(2.0)

    if not results:
        raise RuntimeError("没有任何后端组合完成测试")

    best = max(results, key=lambda item: float(item["end_to_end_fps"]))
    print("\nSUMMARY")
    for result in results:
        print(
            f"{result['capture_requested']}+{result['compute_active']}: "
            f"{result['end_to_end_fps']:>7} FPS "
            f"(compute {result['compute_mean_ms']:>7} ms)"
        )
    print(
        "BEST "
        f"{best['capture_requested']}+{best['compute_active']} "
        f"{best['end_to_end_fps']} FPS"
    )

    report = {
        "capabilities": capabilities.to_dict(),
        "pycuda": pycuda_result,
        "results": results,
        "best": best,
    }
    if args.json is not None:
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON {args.json}")


if __name__ == "__main__":
    main()
