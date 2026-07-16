"""N-channel MJPEG streaming engine with shared JPEG cache."""

import time
import threading
import cv2
import numpy as np
from typing import Dict, Optional


class StreamEngine:
    """Thread-safe multi-channel MJPEG frame cache.

    Features:
    - On-demand encoding: channels with no subscribers skip cv2.imencode
    - Shared cache: multiple viewers of the same channel share one JPEG blob
    - Per-channel FPS throttling
    """

    def __init__(self, max_channels: int = 8):
        self._max_channels = max_channels
        self._channels: Dict[int, dict] = {}
        self._lock = threading.Lock()
        # Initialize all channels
        for i in range(max_channels):
            self._channels[i] = {
                "frame": None,          # np.ndarray or None
                "label": f"通道 {i}",
                "quality": 70,
                "maxfps": 30,
                "last_encoded": None,   # bytes
                "last_ts": 0.0,         # last encode timestamp
                "subscribers": 0,
            }

    def configure(self, channel_id: int, label: str = "",
                  quality: int = 70, maxfps: int = 30):
        """Set channel metadata. Call before streaming."""
        with self._lock:
            ch = self._channels[channel_id]
            if label:
                ch["label"] = label
            ch["quality"] = max(10, min(100, quality))
            ch["maxfps"] = max(1, min(60, maxfps))

    def update(self, channel_id: int, frame: np.ndarray):
        """Push a new frame into the channel (called from main loop)."""
        if channel_id >= self._max_channels:
            return
        with self._lock:
            self._channels[channel_id]["frame"] = frame.copy()

    def subscribe(self, channel_id: int):
        with self._lock:
            self._channels[channel_id]["subscribers"] += 1

    def unsubscribe(self, channel_id: int):
        with self._lock:
            ch = self._channels[channel_id]
            if ch["subscribers"] > 0:
                ch["subscribers"] -= 1

    def get_jpeg(self, channel_id: int) -> Optional[bytes]:
        """Return JPEG bytes for the channel, respecting maxfps and encoding on demand."""
        if channel_id >= self._max_channels:
            return None
        with self._lock:
            ch = self._channels[channel_id]
            if ch["subscribers"] == 0 or ch["frame"] is None:
                return None
            now = time.time()
            min_interval = 1.0 / ch["maxfps"]
            if ch["last_encoded"] is not None and (now - ch["last_ts"]) < min_interval:
                return ch["last_encoded"]
            ret, jpeg = cv2.imencode(
                ".jpg", ch["frame"],
                [int(cv2.IMWRITE_JPEG_QUALITY), ch["quality"]],
            )
            if ret:
                ch["last_encoded"] = jpeg.tobytes()
                ch["last_ts"] = now
                return ch["last_encoded"]
            return ch["last_encoded"]

    def get_frame(self, channel_id: int) -> Optional[np.ndarray]:
        """Get raw frame (for snapshot/recording)."""
        with self._lock:
            return self._channels[channel_id]["frame"]

    def channel_label(self, channel_id: int) -> str:
        with self._lock:
            return self._channels[channel_id]["label"]

    @property
    def channel_count(self) -> int:
        return self._max_channels
