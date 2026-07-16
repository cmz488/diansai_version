"""Camera detection and initialization — migrated from tools.tools.py."""

import os
import glob
import platform
import cv2
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class CameraManager:
    """Detect system cameras and open them with desired config."""

    frame_width: int = 640
    frame_height: int = 480
    frame_fps: str = "61612/513"

    def detect(self) -> List[Dict[str, Any]]:
        """Probe all connected cameras.

        Returns a list of dicts:
            {index, name, default_res, default_fps, supported_resolutions}
        """
        system = platform.system()
        print("=" * 60)
        print(f" 🔍 摄像头探测 (系统: {system})")
        print("=" * 60)

        candidates = self._enumerate(system)
        test_resolutions = [(1920, 1080), (1280, 720), (640, 480), (320, 240)]
        results: List[Dict[str, Any]] = []

        for idx, name in candidates:
            cap = self._open_raw(idx, system)
            if not cap.isOpened():
                continue

            default_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            default_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            default_fps = cap.get(cv2.CAP_PROP_FPS)

            supported = []
            for w, h in test_resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                res = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                if res not in supported:
                    supported.append(res)

            info = {
                "index": idx,
                "name": name,
                "default_res": f"{default_w}x{default_h}",
                "default_fps": f"{default_fps:.1f}" if default_fps > 0 else "未知",
                "supported_resolutions": supported,
            }
            results.append(info)

            print(f"\n[+] 摄像头 [{idx}] {name}")
            print(f"    默认分辨率: {info['default_res']}")
            print(f"    支持分辨率: {', '.join(supported)}")

            cap.release()

        print("\n" + "=" * 60)
        print(f"探测完成，共 {len(results)} 个可用摄像头")
        print("=" * 60)
        return results

    def open(
        self,
        index: int,
        width: int = 1280,
        height: int = 720,
        fps: Optional[float] = None,
    ) -> cv2.VideoCapture:
        """Open a camera and configure it. Returns a ready cv2.VideoCapture."""
        profiles = {
            (320, 240): "2030077/16847",
            (640, 480): "61612/513",
            (1280, 720): "60/1",
            (1920, 1080): "30/1",
        }
        key = (width, height)

        if key not in profiles:
            raise ValueError(f"相机不支持预设分辨率：{width}x{height}")
        system = platform.system()
        self.frame_fps = profiles[key]
        self.frame_width = width
        self.frame_height = height
        if fps is not None:
            actual_fps = float(self.frame_fps.split("/")[0]) / float(
                self.frame_fps.split("/")[1]
            )
            if abs(float(fps) - actual_fps) > 1.0:
                print(
                    f"[Camera] requested {fps:g} fps, using the camera's exact "
                    f"MJPEG profile {self.frame_fps} ({actual_fps:.2f} fps)"
                )
        cap = self._open_raw(index, system)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {index}")
        return cap

    # ── internals ─────────────────────────────────────────────

    def _enumerate(self, system: str) -> List[tuple]:
        """Get (index, name) candidates."""
        if system == "Linux":
            return self._enumerate_linux()
        candidates = [(i, f"Camera {i}") for i in range(16)]
        return candidates

    def _enumerate_linux(self) -> List[tuple]:
        """Linux sysfs scan, filtering IR/metadata virtual nodes."""
        ignore = [
            "metadata",
            "association",
            "statistics",
            "params",
            "meta",
            "ir",
            "depth",
        ]
        paths = sorted(
            glob.glob("/sys/class/video4linux/video*"),
            key=lambda p: int(os.path.basename(p).replace("video", "")),
        )
        result = []
        for p in paths:
            idx = int(os.path.basename(p).replace("video", ""))
            name = "未知摄像头"
            name_file = os.path.join(p, "name")
            if os.path.exists(name_file):
                try:
                    with open(name_file, "r", encoding="utf-8") as f:
                        name = f.read().strip()
                except Exception:
                    pass
            if any(kw in name.lower() for kw in ignore):
                continue
            result.append((idx, name))
        if not result:
            result = [(i, f"Camera {i}") for i in range(16)]
        return result

    def _open_raw(self, index: int, system: str) -> cv2.VideoCapture:
        if system == "Windows":
            return cv2.VideoCapture(index, cv2.CAP_DSHOW)

        pipeline = (
            f"v4l2src device=/dev/video{index} io-mode=mmap ! "
            f"image/jpeg,width={self.frame_width},height={self.frame_height},framerate={self.frame_fps} ! "
            "jpegparse ! "
            "jpegdec ! "
            "video/x-raw,format=BGR ! "
            "appsink sync=false drop=true max-buffers=1"
        )

        if system == "Linux":
            return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        return cv2.VideoCapture(index)
