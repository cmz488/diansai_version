"""Snapshot and video recording for debug sessions."""

import os
import time
import threading
import cv2
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple


class Recorder:
    """Save single-frame snapshots or record frame sequences to video."""

    def __init__(self, save_dir: str = "./photos"):
        self._save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self._recording = False
        self._writer: Optional[cv2.VideoWriter] = None
        self._record_path: Optional[str] = None
        self._lock = threading.Lock()

    # ── snapshot ──────────────────────────────────────────────

    def snapshot(self, frame: np.ndarray) -> str:
        """Save a single frame as JPEG. Returns the filename."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"snapshot_{ts}.jpg"
        filepath = os.path.join(self._save_dir, filename)
        cv2.imwrite(filepath, frame)
        return filename

    # ── recording ─────────────────────────────────────────────

    def start_recording(self, fps: float = 30,
                        size: Tuple[int, int] = (640, 480)) -> bool:
        """Begin recording frames. Call add_frame() in your loop."""
        with self._lock:
            if self._recording:
                return False
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._record_path = os.path.join(self._save_dir, f"record_{ts}.mp4")
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                self._record_path, fourcc, fps, size
            )
            if not self._writer.isOpened():
                self._writer = None
                return False
            self._recording = True
            return True

    def add_frame(self, frame: np.ndarray):
        """Add a frame to the ongoing recording."""
        with self._lock:
            if self._recording and self._writer is not None:
                self._writer.write(frame)

    def stop_recording(self) -> Optional[str]:
        """Stop recording and return the file path."""
        with self._lock:
            self._recording = False
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            return self._record_path

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    # ── gallery ───────────────────────────────────────────────

    def list_files(self) -> List[str]:
        """List saved image/video files, newest first."""
        exts = (".jpg", ".jpeg", ".mp4", ".avi")
        files = [f for f in os.listdir(self._save_dir)
                 if f.lower().endswith(exts)]
        files.sort(reverse=True)
        return files

    def delete_files(self, filenames: List[str]) -> Tuple[List[str], List[str]]:
        """Delete listed files. Returns (deleted, failed)."""
        deleted, failed = [], []
        for f in filenames:
            if ".." in f or "/" in f:
                failed.append(f)
                continue
            fp = os.path.join(self._save_dir, f)
            try:
                if os.path.exists(fp):
                    os.remove(fp)
                    deleted.append(f)
                else:
                    failed.append(f)
            except Exception:
                failed.append(f)
        return deleted, failed
