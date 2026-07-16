"""Thread-safe ring buffer for structured log entries."""

import time
import threading
from typing import Callable, Dict, List, Optional


class LogBuffer:
    """Fixed-size ring buffer of structured log entries.

    Thread-safe. Supports listener callbacks for real-time push.
    """

    def __init__(self, maxlen: int = 500):
        self._maxlen = maxlen
        self._buffer: List[Dict] = []
        self._lock = threading.Lock()
        self._listeners: List[Callable[[Dict], None]] = []

    def append(self, tag: str, msg: str, level: str = "info") -> Dict:
        """Add a log entry. Returns the entry dict."""
        entry = {
            "ts": time.time(),
            "level": level,
            "tag": tag,
            "msg": msg,
        }
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) > self._maxlen:
                self._buffer = self._buffer[-self._maxlen :]
        for cb in list(self._listeners):
            try:
                cb(entry)
            except Exception:
                pass
        return entry

    def snapshot(self) -> List[Dict]:
        """Return all entries (newest last)."""
        with self._lock:
            return list(self._buffer)

    def filter(self, level: Optional[str] = None,
               tag: Optional[str] = None) -> List[Dict]:
        """Return entries matching optional level and/or tag filters."""
        with self._lock:
            result = list(self._buffer)
            if level is not None:
                result = [e for e in result if e["level"] == level]
            if tag is not None:
                result = [e for e in result if e["tag"] == tag]
            return result

    def clear(self):
        """Empty the buffer."""
        with self._lock:
            self._buffer.clear()

    def on_append(self, callback: Callable[[Dict], None]):
        """Register a listener called on each new entry."""
        self._listeners.append(callback)

    def __len__(self):
        with self._lock:
            return len(self._buffer)
