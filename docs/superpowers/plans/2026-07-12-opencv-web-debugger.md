# OpenCV Web Debugger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `tools/web/` — an engineered OpenCV debug & display web tool replacing inline HTML in `tools/tools.py` and `tools/shoot.py`.

**Architecture:** Modular Python backend (7 files) with a template-separated vanilla JS frontend (1 HTML + 1 CSS + 6 JS). Backend provides HTTP REST + MJPEG streams + WebSocket for real-time log/params/metrics push. Frontend is zero-dependency, Apple-white styled, responsive (desktop sidebar + mobile tab bar).

**Tech Stack:** Python 3 stdlib (`http.server`, `threading`, `struct`, `hashlib`, `base64`), OpenCV (`cv2`), vanilla ES6 JS modules, CSS Grid + media queries.

## Global Constraints

- Target: 泰山派 ARM (RK3588-level), must keep CPU/memory overhead low
- Zero JS/CSS framework dependencies (no npm, no CDN)
- Apple white style: `#f5f5f7` bg, `#ffffff` cards, `border-radius: 18px`, SF Pro font stack
- Responsive: desktop sidebar (>768px), mobile tab bar (≤768px), min touch target 44px
- No backward compatibility with old `WebStreamer` / `shoot.py` code
- JPEG encoding only, no PNG
- Shared server-side parameter model (one client's change affects all)
- Python code uses Chinese for user-facing text, English for code identifiers

---

### Task 1: Create directory structure and `__init__.py`

**Files:**
- Create: `tools/web/__init__.py`
- Create: `tools/web/templates/.gitkeep`
- Create: `tools/web/static/css/.gitkeep`
- Create: `tools/web/static/js/.gitkeep`

**Interfaces:**
- Produces: `tools.web.DebugServer`, `tools.web.CameraManager`, `tools.web.ParamRegistry`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p tools/web/templates tools/web/static/css tools/web/static/js
```

- [ ] **Step 2: Write `__init__.py`**

```python
"""OpenCV Web Debugger — engineered debug & display tool."""

from tools.web.server import DebugServer
from tools.web.camera import CameraManager
from tools.web.params import ParamRegistry

__all__ = ["DebugServer", "CameraManager", "ParamRegistry"]
```

- [ ] **Step 3: Create `.gitkeep` files for empty dirs**

```bash
touch tools/web/templates/.gitkeep tools/web/static/css/.gitkeep tools/web/static/js/.gitkeep
```

- [ ] **Step 4: Commit**

```bash
git add tools/web/
git commit -m "feat(web): scaffold tools/web/ directory structure"
```

---

### Task 2: LogBuffer — ring buffer for structured logs

**Files:**
- Create: `tools/web/log_buffer.py`

**Interfaces:**
- Produces: `LogBuffer(maxlen=500)` — `.append(tag, msg, level)`, `.snapshot()`, `.on_append(callback)`

- [ ] **Step 1: Write `log_buffer.py`**

```python
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
        for cb in self._listeners:
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
            result = self._buffer
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
```

- [ ] **Step 2: Commit**

```bash
git add tools/web/log_buffer.py
git commit -m "feat(web): add LogBuffer — thread-safe ring buffer"
```

---

### Task 3: ParamRegistry — parameter registry with validation

**Files:**
- Create: `tools/web/params.py`

**Interfaces:**
- Produces: `ParamRegistry()` — `.add(...)`, `.get(name)`, `.set(name, value)`, `.snapshot()`, `.list_all()`, `.on_change(callback)`

- [ ] **Step 1: Write `params.py`**

```python
"""Parameter registry with type validation and change callbacks."""

import threading
from typing import Any, Callable, Dict, List, Optional


class ParamRegistry:
    """Registry of tunable parameters with validation and callbacks.

    Parameters are shared across all connected browser clients.
    Changes trigger registered callbacks (e.g., WebSocket broadcast).
    """

    VALID_TYPES = ("int", "float", "bool", "choice")

    def __init__(self):
        self._params: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[str, Any], None]] = []

    # ── registration ──────────────────────────────────────────

    def add(
        self,
        name: str,
        type: str = "int",
        default: Any = 0,
        range: Optional[tuple] = None,
        step: Any = None,
        choices: Optional[list] = None,
        group: str = "默认",
        description: str = "",
    ) -> Dict:
        """Register a parameter. Returns the param definition dict."""
        if type not in self.VALID_TYPES:
            raise ValueError(f"Invalid type '{type}', must be one of {self.VALID_TYPES}")
        if type == "choice" and not choices:
            raise ValueError("'choice' type requires a 'choices' list")

        with self._lock:
            self._params[name] = {
                "name": name,
                "type": type,
                "value": default,
                "default": default,
                "range": range,
                "step": step,
                "choices": choices,
                "group": group,
                "description": description,
            }
        return self._params[name]

    # ── read / write ──────────────────────────────────────────

    def get(self, name: str) -> Any:
        """Return the current value of a parameter."""
        with self._lock:
            if name not in self._params:
                raise KeyError(f"Unknown parameter: {name}")
            return self._params[name]["value"]

    def set(self, name: str, value: Any) -> bool:
        """Validate and update a parameter. Returns True on success."""
        with self._lock:
            if name not in self._params:
                return False
            p = self._params[name]
            if not self._validate(p, value):
                return False
            old = p["value"]
            p["value"] = value
        if value != old:
            for cb in self._callbacks:
                try:
                    cb(name, value)
                except Exception:
                    pass
        return True

    def snapshot(self) -> Dict[str, Any]:
        """Return {name: current_value} for all params."""
        with self._lock:
            return {name: p["value"] for name, p in self._params.items()}

    def list_all(self) -> List[Dict]:
        """Return full definitions of all params (for frontend init)."""
        with self._lock:
            return [dict(p) for p in self._params.values()]

    def on_change(self, callback: Callable[[str, Any], None]):
        """Register a callback invoked on every successful param change."""
        self._callbacks.append(callback)

    # ── internals ─────────────────────────────────────────────

    def _validate(self, p: Dict, value: Any) -> bool:
        """Type-check and range-check a single value."""
        t = p["type"]
        try:
            if t == "int":
                value = int(value)
            elif t == "float":
                value = float(value)
            elif t == "bool":
                value = bool(value)
            elif t == "choice":
                if value not in p["choices"]:
                    return False
        except (ValueError, TypeError):
            return False

        if t in ("int", "float") and p["range"] is not None:
            lo, hi = p["range"]
            if not (lo <= value <= hi):
                return False

        return True
```

- [ ] **Step 2: Commit**

```bash
git add tools/web/params.py
git commit -m "feat(web): add ParamRegistry with type validation"
```

---

### Task 4: CameraManager — camera detection and initialization

**Files:**
- Create: `tools/web/camera.py`

**Interfaces:**
- Produces: `CameraManager()` — `.detect()`, `.open(index, width, height, fps, fourcc)`
- Consumes: nothing (standalone, migrates code from `tools/tools.py`)

- [ ] **Step 1: Write `camera.py`**

```python
"""Camera detection and initialization — migrated from tools.tools.py."""

import os
import glob
import platform
import cv2
from typing import Any, Dict, List, Optional


class CameraManager:
    """Detect system cameras and open them with desired config."""

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
        fps: int = 60,
        fourcc: str = "MJPG",
    ) -> cv2.VideoCapture:
        """Open a camera and configure it. Returns a ready cv2.VideoCapture."""
        system = platform.system()
        cap = self._open_raw(index, system)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*fourcc))
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
        ignore = ["metadata", "association", "statistics", "params",
                   "meta", "ir", "depth"]
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
        elif system == "Linux":
            return cv2.VideoCapture(index, cv2.CAP_V4L2)
        return cv2.VideoCapture(index)
```

- [ ] **Step 2: Commit**

```bash
git add tools/web/camera.py
git commit -m "feat(web): add CameraManager — migrate detect_cameras + camera_init"
```

---

### Task 5: StreamEngine — N-channel MJPEG engine

**Files:**
- Create: `tools/web/streamer.py`

**Interfaces:**
- Produces: `StreamEngine(max_channels=8)` — `.configure(channel, ...)`, `.update(channel, frame)`, `.get_jpeg(channel)`, `.subscribe(channel)`, `.unsubscribe(channel)`

- [ ] **Step 1: Write `streamer.py`**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add tools/web/streamer.py
git commit -m "feat(web): add StreamEngine — N-channel MJPEG with shared cache"
```

---

### Task 6: Recorder — snapshot and recording

**Files:**
- Create: `tools/web/recorder.py`

**Interfaces:**
- Produces: `Recorder(save_dir)` — `.snapshot(frame)`, `.start_recording(fps, size)`, `.add_frame(frame)`, `.stop_recording()`, `.list_files()`, `.delete_files(filenames)`

- [ ] **Step 1: Write `recorder.py`**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add tools/web/recorder.py
git commit -m "feat(web): add Recorder — snapshot + MP4 recording"
```

---

### Task 7: DebugServer — HTTP server + WebSocket orchestration

**Files:**
- Create: `tools/web/server.py`

**Interfaces:**
- Produces: `DebugServer(params, port, host, save_dir)` — `.start()`, `.stop()`, `.update_frame(ch, frame)`, `.log(tag, msg, level)`, `.metrics`
- Consumes: `StreamEngine`, `LogBuffer`, `ParamRegistry`, `Recorder`

- [ ] **Step 1: Write `server.py`**

```python
"""DebugServer — HTTP + WebSocket orchestration for the OpenCV debug panel."""

import json
import os
import time
import struct
import hashlib
import base64
import threading
import queue
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingTCPServer
from typing import Any, Callable, Dict, List, Optional, Set

from tools.web.streamer import StreamEngine
from tools.web.log_buffer import LogBuffer
from tools.web.params import ParamRegistry
from tools.web.recorder import Recorder


# ── WebSocket constants ───────────────────────────────────────

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_WS_OP_TEXT = 0x1
_WS_OP_CLOSE = 0x8
_WS_OP_PING = 0x9
_WS_OP_PONG = 0xA


def _ws_accept_key(key: str) -> str:
    return base64.b64encode(
        hashlib.sha1((key + _WS_GUID).encode()).digest()
    ).decode()


def _ws_send(conn, text: str):
    """Send a WebSocket text frame (server→client, unmasked)."""
    payload = text.encode("utf-8")
    length = len(payload)
    frame = bytearray([0x81])  # FIN + text opcode
    if length < 126:
        frame.append(length)
    elif length < 65536:
        frame.append(126)
        frame.extend(struct.pack(">H", length))
    else:
        frame.append(127)
        frame.extend(struct.pack(">Q", length))
    frame.extend(payload)
    conn.sendall(bytes(frame))


def _ws_recv(rfile) -> Optional[str]:
    """Read a WebSocket text frame (client→server, masked). Returns decoded text or None."""
    try:
        b0 = rfile.read(1)
        if not b0:
            return None
        b1 = rfile.read(1)
        if not b1:
            return None
        opcode = b0[0] & 0x0F
        if opcode == _WS_OP_CLOSE:
            return None
        if opcode == _WS_OP_PING:
            # Read and discard ping payload, pong handled by caller
            length = b1[0] & 0x7F
            if length == 126:
                length = struct.unpack(">H", rfile.read(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", rfile.read(8))[0]
            if length > 0:
                rfile.read(length)
            return None  # caller checks opcode

        masked = b1[0] & 0x80
        length = b1[0] & 0x7F
        if length == 126:
            length = struct.unpack(">H", rfile.read(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", rfile.read(8))[0]
        if length > 1024 * 1024:  # 1 MB limit
            return None
        if masked:
            mask = rfile.read(4)
            payload = bytearray(rfile.read(length))
            for i in range(len(payload)):
                payload[i] ^= mask[i % 4]
            return bytes(payload).decode("utf-8")
        else:
            return rfile.read(length).decode("utf-8")
    except Exception:
        return None


# ── Metrics collector ─────────────────────────────────────────

class MetricsCollector:
    """Lightweight key-value metrics store."""

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def update(self, **kv):
        with self._lock:
            self._data.update(kv)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)


# ── Static file serving ───────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE_DIR = os.path.join(_HERE, "templates")
_STATIC_DIR = os.path.join(_HERE, "static")

_MIME_MAP = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _serve_file(path: str, handler) -> bool:
    """Serve a static file. Returns True if found."""
    if ".." in path:
        handler.send_error(403)
        return True
    full = os.path.join(_STATIC_DIR, path.lstrip("/"))
    if not os.path.isfile(full):
        return False
    ext = os.path.splitext(full)[1].lower()
    mime = _MIME_MAP.get(ext, "application/octet-stream")
    with open(full, "rb") as f:
        data = f.read()
    handler.send_response(200)
    handler.send_header("Content-Type", mime)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)
    return True


# ── HTTP request handler ──────────────────────────────────────

class _DebugHandler(BaseHTTPRequestHandler):
    """Routes HTTP requests to DebugServer methods."""

    def log_message(self, fmt, *args):
        pass  # silent

    def do_GET(self):
        ds: "DebugServer" = self.server.debug_server

        if self.path == "/":
            return self._serve_template()

        if self.path.startswith("/static/"):
            path = self.path[len("/static"):]
            if _serve_file(path, self):
                return
            self.send_error(404)
            return

        if self.path.startswith("/stream/"):
            return self._serve_stream()

        if self.path == "/ws":
            return self._ws_handler()

        if self.path == "/api/params":
            return self._json(ds.params.list_all())

        if self.path == "/api/cameras":
            return self._json(ds._camera_list_cache)

        if self.path == "/api/gallery":
            return self._json(ds.recorder.list_files())

        if self.path == "/api/metrics":
            return self._json(ds.metrics.snapshot())

        if self.path.startswith("/preview/"):
            return self._serve_preview()

        self.send_error(404)

    def do_POST(self):
        ds: "DebugServer" = self.server.debug_server
        body = self._read_body()

        if self.path == "/api/params":
            name = body.get("name")
            value = body.get("value")
            if name is None or value is None:
                self.send_error(400, "Missing name or value")
                return
            ok = ds.params.set(name, value)
            if ok:
                ds._broadcast_params()
            self._json({"ok": ok, "params": ds.params.snapshot()})
            return

        if self.path == "/api/cameras/active":
            idx = body.get("index")
            self._json({"ok": False, "message": "Camera switching not in this version"})
            return

        if self.path == "/api/snapshot":
            frame = ds.streamer.get_frame(0)
            if frame is None:
                self._json({"success": False, "message": "无可用帧"})
                return
            fname = ds.recorder.snapshot(frame)
            self._json({"success": True, "filename": fname})
            return

        if self.path == "/api/recording":
            action = body.get("action", "")
            if action == "start":
                ok = ds.recorder.start_recording()
                self._json({"success": ok, "recording": ok})
            elif action == "stop":
                path = ds.recorder.stop_recording()
                self._json({"success": True, "path": path})
            else:
                self.send_error(400, "Unknown action")
            return

        if self.path == "/api/photos":
            action = body.get("action", "")
            if action == "delete":
                files = body.get("files", [])
                deleted, failed = ds.recorder.delete_files(files)
                self._json({"success": True, "deleted": deleted, "failed": failed})
                return
            self.send_error(400)
            return

        self.send_error(404)

    # ── helpers ───────────────────────────────────────────────

    def _serve_template(self):
        tp = os.path.join(_TEMPLATE_DIR, "index.html")
        if os.path.isfile(tp):
            with open(tp, "r", encoding="utf-8") as f:
                html = f.read()
        else:
            html = "<html><body><h1>模板未找到</h1></body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _serve_stream(self):
        try:
            ch = int(self.path.split("/")[-1])
        except ValueError:
            self.send_error(400)
            return

        ds: "DebugServer" = self.server.debug_server
        ds.streamer.subscribe(ch)

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                jpeg = ds.streamer.get_jpeg(ch)
                if jpeg is not None:
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                time.sleep(0.03)
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            ds.streamer.unsubscribe(ch)

    def _serve_preview(self):
        filename = self.path.split("/")[-1]
        if ".." in filename or "/" in filename:
            self.send_error(403)
            return
        ds: "DebugServer" = self.server.debug_server
        filepath = os.path.join(ds.recorder._save_dir, filename)
        if not os.path.exists(filepath):
            self.send_error(404)
            return
        with open(filepath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _ws_handler(self):
        """Perform WebSocket upgrade and enter frame loop."""
        ds: "DebugServer" = self.server.debug_server
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self.send_error(400)
            return

        self.send_response(101)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", _ws_accept_key(key))
        self.end_headers()

        # Get underlying socket for raw WebSocket writes
        conn = self.request

        ds._ws_add(conn)
        try:
            while True:
                msg = _ws_recv(self.rfile)
                if msg is None:
                    break
                ds._ws_dispatch(msg)
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            ds._ws_remove(conn)

    def _json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            return json.loads(self.rfile.read(length))
        return {}


# ── DebugServer ───────────────────────────────────────────────

class DebugServer:
    """Top-level orchestrator for the OpenCV web debug panel.

    Usage:
        server = DebugServer(params=params, port=8080)
        server.start()  # non-blocking

        while True:
            ...
            server.update_frame(0, frame)
            server.log("DETECT", "found target")
            server.metrics.update(fps=58, detect=12)
    """

    def __init__(
        self,
        params: Optional[ParamRegistry] = None,
        port: int = 8080,
        host: str = "0.0.0.0",
        save_dir: str = "./photos",
    ):
        self.params = params or ParamRegistry()
        self.streamer = StreamEngine(max_channels=8)
        self.log_buffer = LogBuffer(maxlen=500)
        self.recorder = Recorder(save_dir)
        self.metrics = MetricsCollector()

        self._port = port
        self._host = host
        self._server: Optional[ThreadingTCPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # WebSocket client set
        self._ws_clients: Set[Any] = set()
        self._ws_lock = threading.Lock()

        # Camera list cache (populated externally or by detect)
        self._camera_list_cache: List[Dict] = []

        # Metrics broadcast timer
        self._metrics_interval = 1.0
        self._metrics_last = 0.0

        # Hook log_buffer → WS broadcast
        self.log_buffer.on_append(self._on_log_entry)

    # ── lifecycle ─────────────────────────────────────────────

    def start(self):
        """Start HTTP + WS server in a background daemon thread."""
        if self._running:
            return

        _DebugHandler.server = None  # will be set below

        class _Server(ThreadingTCPServer, HTTPServer):
            allow_reuse_address = True

        self._server = _Server((self._host, self._port), _DebugHandler)
        self._server.debug_server = self

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._running = True

        print(f"\n[DebugServer] http://{self._host}:{self._port}")
        print(f"[DebugServer] WebSocket ws://{self._host}:{self._port}/ws\n")

    def stop(self):
        """Stop the server."""
        self._running = False
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        print("[DebugServer] 已关闭")

    # ── public API ────────────────────────────────────────────

    def update_frame(self, channel_id: int, frame):
        """Push a frame to a stream channel."""
        self.streamer.update(channel_id, frame)
        # If recording, write to recorder
        if self.recorder.is_recording and channel_id == 0:
            self.recorder.add_frame(frame)

    def log(self, tag: str, msg: str, level: str = "info"):
        """Append a structured log entry (broadcast to WS clients)."""
        return self.log_buffer.append(tag, msg, level)

    # ── WebSocket internals ───────────────────────────────────

    def _ws_add(self, conn):
        with self._ws_lock:
            self._ws_clients.add(conn)
        # Send current params + recent logs on connect
        self._ws_send(conn, {
            "type": "params",
            "params": self.params.snapshot(),
        })

    def _ws_remove(self, conn):
        with self._ws_lock:
            self._ws_clients.discard(conn)

    def _ws_dispatch(self, raw: str):
        """Handle incoming WS message."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        if msg.get("type") == "set_param":
            name = msg.get("name")
            value = msg.get("value")
            if name is not None and value is not None:
                self.params.set(name, value)
                self._broadcast_params()
        elif msg.get("type") == "ping":
            pass  # pong handled by _ws_recv returning None for ping

    def _broadcast_params(self):
        data = json.dumps({
            "type": "params",
            "params": self.params.snapshot(),
        }, ensure_ascii=False)
        self._broadcast_raw(data)

    def _ws_send(self, conn, obj: dict):
        try:
            _ws_send(conn, json.dumps(obj, ensure_ascii=False))
        except Exception:
            self._ws_remove(conn)

    def _broadcast_raw(self, text: str):
        """Send text to all WS clients."""
        dead = set()
        with self._ws_lock:
            for conn in self._ws_clients:
                try:
                    _ws_send(conn, text)
                except Exception:
                    dead.add(conn)
            self._ws_clients -= dead

    def _on_log_entry(self, entry: Dict):
        """Called by LogBuffer on each new entry — push to WS."""
        self._broadcast_raw(json.dumps({"type": "log", **entry}, ensure_ascii=False))

    def broadcast_metrics(self):
        """Broadcast current metrics snapshot to all WS clients (call periodically)."""
        now = time.time()
        if now - self._metrics_last < self._metrics_interval:
            return
        self._metrics_last = now
        data = json.dumps({
            "type": "metrics",
            **self.metrics.snapshot(),
        }, ensure_ascii=False)
        self._broadcast_raw(data)
```

- [ ] **Step 2: Commit**

```bash
git add tools/web/server.py
git commit -m "feat(web): add DebugServer — HTTP routes + WebSocket orchestration"
```

---

### Task 8: debug.css — Apple white responsive styles

**Files:**
- Create: `tools/web/static/css/debug.css`

**Interfaces:**
- Produces: CSS classes for layout, components, and responsive breakpoints
- Consumes: nothing

- [ ] **Step 1: Write `debug.css`**

```css
/* ── Reset & Base ─────────────────────────────────────────── */
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

:root {
  --bg: #f5f5f7;
  --card: #ffffff;
  --text: #1d1d1f;
  --text-secondary: #86868b;
  --accent: #0071e3;
  --accent-light: #e8f0fe;
  --border: #d2d2d7;
  --shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
  --radius: 18px;
  --radius-sm: 10px;
  --font: -apple-system, "SF Pro Display", "PingFang SC", "Microsoft YaHei", sans-serif;
  --sidebar-w: 260px;
  --titlebar-h: 52px;
  --statusbar-h: 40px;
  --tabbar-h: 52px;
  --danger: #ff3b30;
  --danger-bg: #ffe5e3;
  --success: #34c759;
  --warn: #ff9f0a;
}

body {
  font-family: var(--font);
  background: var(--bg);
  color: var(--text);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  overflow: hidden;
  height: 100vh;
  display: flex;
  flex-direction: column;
}

/* ── Title Bar ────────────────────────────────────────────── */
.titlebar {
  height: var(--titlebar-h);
  background: rgba(255, 255, 255, 0.72);
  backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 20px;
  gap: 12px;
  z-index: 100;
  flex-shrink: 0;
}
.titlebar .dots { display: flex; gap: 6px; }
.titlebar .dot {
  width: 12px; height: 12px; border-radius: 50%;
}
.titlebar .dot.r { background: #ff5f57; }
.titlebar .dot.y { background: #febc2e; }
.titlebar .dot.g { background: #28c840; }
.titlebar .title {
  font-size: 15px; font-weight: 700; flex: 1; text-align: center;
  color: var(--text);
}
.titlebar .ip {
  font-size: 12px; color: var(--text-secondary);
}

/* ── Main Layout ──────────────────────────────────────────── */
.main { display: flex; flex: 1; overflow: hidden; }

/* Sidebar */
.sidebar {
  width: var(--sidebar-w);
  background: var(--card);
  border-right: 1px solid var(--border);
  overflow-y: auto;
  flex-shrink: 0;
  padding: 16px 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.sidebar-section { margin-bottom: 4px; }
.sidebar-section > .label {
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  color: var(--text-secondary); letter-spacing: 0.5px;
  padding: 8px 8px 4px;
}
.sidebar-section > .content { padding: 0 4px; }

/* ── Viewport (stream area) ───────────────────────────────── */
.viewport {
  flex: 1; overflow-y: auto; padding: 16px;
  display: flex; align-items: flex-start; justify-content: center;
}
.grid {
  display: grid;
  gap: 12px;
  width: 100%;
  max-width: 1400px;
}
.grid.cols1 { grid-template-columns: 1fr; }
.grid.cols2 { grid-template-columns: repeat(2, 1fr); }
.grid.cols4 { grid-template-columns: repeat(2, 1fr); }

.stream-cell {
  background: #000;
  border-radius: var(--radius-sm);
  overflow: hidden;
  border: 1px solid var(--border);
  position: relative;
  aspect-ratio: 4/3;
}
.stream-cell img {
  width: 100%; height: 100%; object-fit: contain; display: block;
}
.stream-cell .cell-label {
  position: absolute; top: 6px; left: 8px;
  background: rgba(0,0,0,0.55); color: #fff;
  font-size: 11px; padding: 2px 10px; border-radius: 6px;
  font-weight: 600; pointer-events: none;
}

/* ── Status Bar ───────────────────────────────────────────── */
.statusbar {
  height: var(--statusbar-h);
  background: var(--card);
  border-top: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 16px;
  gap: 24px; font-size: 12px; color: var(--text-secondary);
  flex-shrink: 0;
}
.statusbar .metric { display: flex; align-items: center; gap: 6px; }
.statusbar .metric .val { font-weight: 700; color: var(--text); }
.statusbar .rec-dot {
  width: 8px; height: 8px; border-radius: 50%; background: var(--danger);
  display: none;
}
.statusbar .rec-dot.active { display: inline-block; animation: pulse 1s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Bottom Panels ────────────────────────────────────────── */
.panels { border-top: 1px solid var(--border); flex-shrink: 0; }
.panel-toggle {
  display: flex; gap: 4px; padding: 6px 16px;
  background: var(--bg); border-bottom: 1px solid var(--border);
}
.panel-toggle button {
  background: none; border: none; padding: 6px 14px;
  font-size: 12px; font-weight: 600; color: var(--text-secondary);
  cursor: pointer; border-radius: 6px; font-family: var(--font);
  transition: all 0.15s;
}
.panel-toggle button:hover { background: var(--accent-light); color: var(--accent); }
.panel-toggle button.active { background: var(--accent); color: #fff; }

.panel-chart, .panel-log {
  height: 160px; overflow: hidden; display: none;
}
.panel-chart.show, .panel-log.show { display: block; }
.panel-chart canvas { width: 100%; height: 100%; }
#logList {
  height: 100%; overflow-y: auto; padding: 4px 16px;
  font-size: 11px; font-family: "SF Mono", "Menlo", "Consolas", monospace;
}
.log-row { padding: 2px 0; border-bottom: 1px solid #f0f0f0;
  display: flex; gap: 8px; }
.log-row .log-ts { color: var(--text-secondary); flex-shrink: 0; }
.log-row .log-tag { font-weight: 700; flex-shrink: 0; }
.log-row.log-info  .log-tag { color: var(--accent); }
.log-row.log-warn  .log-tag { color: var(--warn); }
.log-row.log-error .log-tag { color: var(--danger); }
.log-row .log-msg { color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── Parameter Controls ───────────────────────────────────── */
.param-row { margin-bottom: 10px; }
.param-row .param-label {
  display: flex; justify-content: space-between;
  font-size: 12px; margin-bottom: 4px; color: var(--text);
}
.param-row .param-label .val { font-weight: 700; color: var(--text-secondary); }
.param-row input[type="range"] {
  width: 100%; height: 6px;
  -webkit-appearance: none; appearance: none;
  background: #e0e0e0; border-radius: 3px; outline: none;
}
.param-row input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none;
  width: 20px; height: 20px;
  border-radius: 50%; background: #fff;
  border: 2px solid var(--accent);
  box-shadow: 0 1px 4px rgba(0,0,0,0.1);
  cursor: pointer;
}
.param-row input[type="checkbox"] {
  width: 20px; height: 20px; accent-color: var(--accent);
}
.param-row select, .param-row button {
  font-family: var(--font); font-size: 12px;
  padding: 6px 12px; border-radius: 8px;
  border: 1px solid var(--border); background: var(--card);
  cursor: pointer;
}
.param-row button { font-weight: 600; }
.param-row button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
.param-row button.danger { background: var(--danger-bg); color: var(--danger); border-color: var(--danger); }

/* ── ROI Overlay ──────────────────────────────────────────── */
.stream-cell canvas.roi-overlay {
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  cursor: crosshair; display: none;
}
.stream-cell canvas.roi-overlay.active { display: block; }
.roi-stats {
  position: absolute; bottom: 4px; right: 8px;
  background: rgba(0,0,0,0.7); color: #fff;
  font-size: 11px; padding: 4px 10px; border-radius: 6px;
  pointer-events: none;
}

/* ── Tab Bar (mobile) ─────────────────────────────────────── */
.tabbar {
  display: none; height: var(--tabbar-h);
  background: rgba(255,255,255,0.92);
  backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
  border-top: 1px solid var(--border);
  justify-content: space-around; align-items: center;
  flex-shrink: 0; z-index: 100;
}
.tabbar button {
  background: none; border: none; font-size: 10px; font-weight: 600;
  color: var(--text-secondary); display: flex; flex-direction: column;
  align-items: center; gap: 3px; cursor: pointer; font-family: var(--font);
  padding: 4px 12px;
}
.tabbar button .icon { font-size: 20px; }
.tabbar button.active { color: var(--accent); }

/* ── Mobile Overlays ──────────────────────────────────────── */
.mobile-panel {
  display: none; position: fixed; inset: 0; z-index: 200;
  background: var(--bg); padding: 20px; overflow-y: auto;
  padding-bottom: 80px;
}
.mobile-panel.show { display: block; }
.mobile-panel .close-btn {
  float: right; font-size: 24px; background: none; border: none;
  cursor: pointer; color: var(--text-secondary); line-height: 1;
}

/* ── Responsive Breakpoint ────────────────────────────────── */
@media (max-width: 768px) {
  .sidebar { display: none; }
  .tabbar { display: flex; }
  .grid.cols2, .grid.cols4 { grid-template-columns: 1fr; }
  .panels { display: none; }
  .statusbar { font-size: 11px; gap: 12px; }
}

/* ── Utils ────────────────────────────────────────────────── */
.toggle-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 8px; font-size: 12px;
}
```

- [ ] **Step 2: Commit**

```bash
git add tools/web/static/css/debug.css
git commit -m "feat(web): add Apple-white responsive CSS"
```

---

### Task 9: JS utilities + log + chart components

**Files:**
- Create: `tools/web/static/js/utils.js`
- Create: `tools/web/static/js/log.js`
- Create: `tools/web/static/js/chart.js`

- [ ] **Step 1: Write `utils.js`**

```javascript
/** DOM utilities and debounce helper. */

export function $(sel, ctx = document) { return ctx.querySelector(sel); }
export function $$(sel, ctx = document) { return [...ctx.querySelectorAll(sel)]; }

export function debounce(fn, ms = 50) {
  let id;
  return (...args) => { clearTimeout(id); id = setTimeout(() => fn(...args), ms); };
}

export function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("zh-CN", { hour12: false });
}
```

- [ ] **Step 2: Write `log.js`**

```javascript
/** Virtual-scrolling log panel connected to WebSocket. */

import { $, fmtTime } from "./utils.js";

const MAX_LINES = 500;
const VISIBLE = 20;
let entries = [];
let ws = null;

export function initLog(wsInstance) {
  ws = wsInstance;
}

export function handleLogMessage(data) {
  entries.push(data);
  if (entries.length > MAX_LINES) entries = entries.slice(-MAX_LINES);
  render();
}

function render() {
  const list = $("#logList");
  if (!list || !list.parentElement.classList.contains("show")) return;

  const start = Math.max(0, entries.length - VISIBLE);
  const html = entries.slice(start).map(e => {
    const cls = `log-row log-${e.level}`;
    return `<div class="${cls}">
      <span class="log-ts">${fmtTime(e.ts)}</span>
      <span class="log-tag">[${e.tag}]</span>
      <span class="log-msg">${esc(e.msg)}</span>
    </div>`;
  }).join("");
  list.innerHTML = html;
  list.scrollTop = list.scrollHeight;
}

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** Force re-render (called when panel becomes visible or on resize). */
export function refreshLog() { render(); }
```

- [ ] **Step 3: Write `chart.js`**

```javascript
/** Lightweight Canvas 2D line chart with incremental drawing.

 *  Uses shift-draw technique: moves existing pixels left, draws
 *  only the new data segment on the right edge.
 */

const MAX_POINTS = 120;

export function initChart(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;

  const ctx = canvas.getContext("2d");
  const series = {};   // { key: { color, data: [] } }

  function resize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + "px";
    canvas.style.height = rect.height + "px";
    ctx.setTransform(1, 0, 0, 1, 0, 0); // reset
    ctx.scale(dpr, dpr);
    redrawAll();
  }

  function addSeries(key, color) {
    series[key] = { color, data: [] };
  }

  function push(key, value) {
    const s = series[key];
    if (!s) return;
    s.data.push(value);
    if (s.data.length > MAX_POINTS) s.data.shift();
    drawIncremental();
  }

  function drawIncremental() {
    const w = canvas.width / (window.devicePixelRatio || 1);
    const h = canvas.height / (window.devicePixelRatio || 1);
    const step = w / MAX_POINTS;

    // Shift existing content left
    ctx.drawImage(canvas,
      step * (window.devicePixelRatio || 1), 0,
      w * (window.devicePixelRatio || 1) - step * (window.devicePixelRatio || 1),
      h * (window.devicePixelRatio || 1),
      0, 0,
      w * (window.devicePixelRatio || 1) - step * (window.devicePixelRatio || 1),
      h * (window.devicePixelRatio || 1));
    ctx.clearRect(w - step, 0, step, h);

    // Find global min/max across all series for Y scale
    let gmin = Infinity, gmax = -Infinity;
    for (const s of Object.values(series)) {
      for (const v of s.data) {
        if (v < gmin) gmin = v;
        if (v > gmax) gmax = v;
      }
    }
    if (!isFinite(gmin)) { gmin = 0; gmax = 100; }
    const range = gmax - gmin || 1;
    const pad = range * 0.1;

    // Draw last point + line segment for each series
    for (const s of Object.values(series)) {
      if (s.data.length < 2) continue;
      const len = s.data.length;
      const x1 = w - step * 2;
      const y1 = h - ((s.data[len - 2] - gmin + pad) / (range + pad * 2)) * h;
      const x2 = w - step;
      const y2 = h - ((s.data[len - 1] - gmin + pad) / (range + pad * 2)) * h;

      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(Math.max(0, x1), y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  }

  function redrawAll() {
    // Full redraw (called on resize or series add)
    const w = canvas.width / (window.devicePixelRatio || 1);
    const h = canvas.height / (window.devicePixelRatio || 1);
    ctx.clearRect(0, 0, w, h);

    let gmin = Infinity, gmax = -Infinity;
    for (const s of Object.values(series)) {
      for (const v of s.data) { if (v < gmin) gmin = v; if (v > gmax) gmax = v; }
    }
    if (!isFinite(gmin)) { gmin = 0; gmax = 100; }
    const range = gmax - gmin || 1;
    const pad = range * 0.1;
    const step = w / MAX_POINTS;

    for (const s of Object.values(series)) {
      if (s.data.length < 2) continue;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < s.data.length; i++) {
        const x = (i / MAX_POINTS) * w;
        const y = h - ((s.data[i] - gmin + pad) / (range + pad * 2)) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }

  window.addEventListener("resize", resize);
  resize();

  return { addSeries, push, resize: () => resize() };
}
```

- [ ] **Step 4: Commit**

```bash
git add tools/web/static/js/utils.js tools/web/static/js/log.js tools/web/static/js/chart.js
git commit -m "feat(web): add JS utils, virtual-scroll log, canvas chart"
```

---

### Task 10: JS params + layout + roi components

**Files:**
- Create: `tools/web/static/js/params.js`
- Create: `tools/web/static/js/layout.js`
- Create: `tools/web/static/js/roi.js`

- [ ] **Step 1: Write `params.js`**

```javascript
/** Parameter panel — renders sliders/toggles/dropdowns, POSTs changes. */

import { $, $$, debounce } from "./utils.js";

let paramDefs = [];
let paramValues = {};
let currentGroup = "all";

const POST = debounce(async (name, value) => {
  await fetch("/api/params", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, value }),
  });
}, 50);

export async function loadParams() {
  const r = await fetch("/api/params");
  paramDefs = await r.json();
  for (const p of paramDefs) paramValues[p.name] = p.value;
  render();
}

export function handleParamsMessage(data) {
  paramValues = { ...paramValues, ...data.params };
  renderValues();
}

function render() {
  const container = $("#paramList");
  if (!container) return;

  const groups = {};
  for (const p of paramDefs) {
    (groups[p.group] ||= []).push(p);
  }

  let html = "";
  for (const [group, params] of Object.entries(groups)) {
    html += `<div class="sidebar-section">
      <div class="label">${group}</div>
      <div class="content">`;
    for (const p of params) {
      const val = paramValues[p.name] ?? p.default;
      html += renderParamRow(p, val);
    }
    html += `</div></div>`;
  }
  container.innerHTML = html;

  // Bind events
  for (const p of paramDefs) {
    const el = $(`[data-param="${p.name}"]`);
    if (!el) continue;
    el.addEventListener("input", () => {
      const v = p.type === "bool" ? el.checked : el.value;
      paramValues[p.name] = p.type === "int" ? parseInt(v)
        : p.type === "float" ? parseFloat(v) : v;
      $(`[data-param-val="${p.name}"]`).textContent = paramValues[p.name];
      POST(p.name, paramValues[p.name]);
    });
  }
}

function renderParamRow(p, val) {
  if (p.type === "bool") {
    return `<div class="toggle-row">
      <span>${p.name}</span>
      <input type="checkbox" data-param="${p.name}" ${val ? "checked" : ""} />
    </div>`;
  }
  if (p.type === "choice") {
    let opts = (p.choices || []).map(c =>
      `<option value="${c}" ${c === val ? "selected" : ""}>${c}</option>`
    ).join("");
    return `<div class="param-row">
      <div class="param-label"><span>${p.name}</span></div>
      <select data-param="${p.name}">${opts}</select>
    </div>`;
  }
  // int / float slider
  const [lo, hi] = p.range || [0, 100];
  const step = p.step ?? (p.type === "float" ? 0.1 : 1);
  return `<div class="param-row">
    <div class="param-label">
      <span>${p.name}</span>
      <span class="val" data-param-val="${p.name}">${val}</span>
    </div>
    <input type="range" data-param="${p.name}"
      min="${lo}" max="${hi}" step="${step}" value="${val}" />
  </div>`;
}

function renderValues() {
  // Update displayed values without rebuilding DOM
  for (const p of paramDefs) {
    const el = $(`[data-param="${p.name}"]`);
    const label = $(`[data-param-val="${p.name}"]`);
    if (!el) continue;
    const val = paramValues[p.name] ?? p.default;
    if (p.type === "bool") el.checked = !!val;
    else if (p.type !== "choice") {
      el.value = val;
      if (label) label.textContent = val;
    }
  }
}
```

- [ ] **Step 2: Write `layout.js`**

```javascript
/** Multi-view grid layout manager with responsive mobile swipe. */

import { $, $$ } from "./utils.js";

let gridMode = "cols2";  // cols1 | cols2 | cols4
let streamChannels = [0, 1, 2, 3]; // which channel per cell
let currentMobileIdx = 0;

export function setGrid(mode) {
  gridMode = mode;
  const grid = $("#streamGrid");
  grid.className = "grid " + mode;

  const count = mode === "cols1" ? 1 : mode === "cols2" ? 2 : 4;
  renderCells(count);
}

export function setChannel(cellIdx, channelId) {
  streamChannels[cellIdx] = channelId;
  const count = gridMode === "cols1" ? 1 : gridMode === "cols2" ? 2 : 4;
  renderCells(count);
}

function renderCells(count) {
  const grid = $("#streamGrid");
  grid.innerHTML = "";
  for (let i = 0; i < count; i++) {
    const ch = streamChannels[i] ?? i;
    const cell = document.createElement("div");
    cell.className = "stream-cell";
    cell.innerHTML = `
      <div class="cell-label">通道 ${ch}</div>
      <img src="/stream/${ch}" alt="通道 ${ch}" data-stream="${ch}" />
    `;
    grid.appendChild(cell);
  }
}

export function initMobileSwipe() {
  const viewport = $(".viewport");
  if (!viewport) return;
  let startX = 0;

  viewport.addEventListener("touchstart", e => {
    startX = e.touches[0].clientX;
  }, { passive: true });

  viewport.addEventListener("touchend", e => {
    const dx = e.changedTouches[0].clientX - startX;
    if (Math.abs(dx) < 40) return;
    const count = gridMode === "cols1" ? 1 : gridMode === "cols2" ? 2 : 4;
    if (dx < 0) currentMobileIdx = Math.min(currentMobileIdx + 1, count - 1);
    else currentMobileIdx = Math.max(currentMobileIdx - 1, 0);
    // Show only current cell on mobile
    $$(".stream-cell").forEach((c, i) => {
      c.style.display = (window.innerWidth <= 768 && i !== currentMobileIdx)
        ? "none" : "";
    });
  });

  // Handle dot indicators (mobile)
  window.addEventListener("resize", () => {
    if (window.innerWidth > 768) {
      $$(".stream-cell").forEach(c => c.style.display = "");
    } else {
      updateMobileVisible();
    }
  });
}

function updateMobileVisible() {
  const count = gridMode === "cols1" ? 1 : gridMode === "cols2" ? 2 : 4;
  $$(".stream-cell").forEach((c, i) => {
    c.style.display = (i === currentMobileIdx || count <= 1) ? "" : "none";
  });
}

/** Suspend non-visible stream img src to save decode CPU on ARM. */
export function throttleStreams() {
  const visible = window.innerWidth <= 768 ? 1 : 2;
  $$(".stream-cell img").forEach((img, i) => {
    if (i >= visible) {
      if (img.dataset.srcSaved === undefined) {
        img.dataset.srcSaved = img.src;
        img.src = "";
      }
    } else if (img.dataset.srcSaved) {
      img.src = img.dataset.srcSaved;
      delete img.dataset.srcSaved;
    }
  });
}
```

- [ ] **Step 3: Write `roi.js`**

```javascript
/** ROI selection overlay on stream images. */

let active = false;
let roiCanvas = null;
let roiCtx = null;
let drawing = false;
let startX, startY, endX, endY;

export function initROI(canvasId) {
  roiCanvas = document.getElementById(canvasId);
  if (!roiCanvas) return;
  roiCtx = roiCanvas.getContext("2d");

  roiCanvas.addEventListener("pointerdown", e => {
    if (!active) return;
    drawing = true;
    const rect = roiCanvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
  });

  roiCanvas.addEventListener("pointermove", e => {
    if (!active || !drawing) return;
    const rect = roiCanvas.getBoundingClientRect();
    endX = e.clientX - rect.left;
    endY = e.clientY - rect.top;
    drawRect();
  });

  roiCanvas.addEventListener("pointerup", () => {
    if (!active || !drawing) return;
    drawing = false;
    showStats();
  });
}

export function toggleROI() {
  active = !active;
  roiCanvas.classList.toggle("active", active);
  if (!active) roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
}

function drawRect() {
  const w = endX - startX;
  const h = endY - startY;
  roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
  roiCtx.strokeStyle = "#0071e3";
  roiCtx.lineWidth = 2;
  roiCtx.setLineDash([6, 3]);
  roiCtx.strokeRect(startX, startY, w, h);
  roiCtx.setLineDash([]);
}

function showStats() {
  // Show pixel stats in the corner
  const w = Math.abs(endX - startX);
  const h = Math.abs(endY - startY);
  let el = document.querySelector(".roi-stats");
  if (!el) {
    el = document.createElement("div");
    el.className = "roi-stats";
    roiCanvas.parentElement.appendChild(el);
  }
  el.textContent = `${Math.round(w)}×${Math.round(h)} | 选区`;
  el.style.display = "";
  setTimeout(() => { el.style.display = "none"; }, 3000);
}
```

- [ ] **Step 4: Commit**

```bash
git add tools/web/static/js/params.js tools/web/static/js/layout.js tools/web/static/js/roi.js
git commit -m "feat(web): add params panel, grid layout, ROI overlay"
```

---

### Task 11: App controller + HTML template

**Files:**
- Create: `tools/web/static/js/app.js`
- Create: `tools/web/templates/index.html`

- [ ] **Step 1: Write `app.js`**

```javascript
/** Main application controller — WebSocket, tabs, orchestration. */

import { $ } from "./utils.js";
import { initLog, handleLogMessage, refreshLog } from "./log.js";
import { initChart } from "./chart.js";
import { loadParams, handleParamsMessage } from "./params.js";
import { setGrid, initMobileSwipe, throttleStreams } from "./layout.js";
import { initROI, toggleROI } from "./roi.js";

let ws;
let chart;

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onmessage = e => {
    const msg = JSON.parse(e.data);
    switch (msg.type) {
      case "log":     handleLogMessage(msg); break;
      case "params":  handleParamsMessage(msg); break;
      case "metrics": updateMetrics(msg); break;
    }
  };

  ws.onclose = () => setTimeout(connectWS, 2000);
}

function updateMetrics(data) {
  const { fps, detect, latency } = data;
  if (fps !== undefined) {
    const el = $("#metricFps"); if (el) el.textContent = fps;
    if (chart) chart.push("fps", fps);
  }
  if (detect !== undefined) {
    const el = $("#metricDetect"); if (el) el.textContent = detect;
    if (chart) chart.push("detect", detect);
  }
  if (latency !== undefined) {
    const el = $("#metricLatency"); if (el) el.textContent = latency.toFixed(1) + "ms";
  }
}

// ── Tab switching ──────────────────────────────────────────
function initTabs() {
  // Desktop: sidebar visibility via toggle buttons in sidebar-section labels
  // Mobile: bottom tab bar
  const tabs = document.querySelectorAll(".tabbar button");
  tabs.forEach(btn => {
    btn.addEventListener("click", () => {
      tabs.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const panel = btn.dataset.panel;
      showMobilePanel(panel);
    });
  });

  // Panel toggle (chart/log) on desktop
  $("#btnChart")?.addEventListener("click", () => toggleBottomPanel("chart"));
  $("#btnLog")?.addEventListener("click", () => toggleBottomPanel("log"));
}

function showMobilePanel(name) {
  document.querySelectorAll(".mobile-panel").forEach(p => p.classList.remove("show"));
  const panel = $(`#mobile-${name}`);
  if (panel) {
    panel.classList.add("show");
    if (name === "log") refreshLog();
  }
}

function toggleBottomPanel(name) {
  const el = $(`.panel-${name}`);
  if (!el) return;
  const show = !el.classList.contains("show");
  el.classList.toggle("show", show);
  const btn = $(`#btn${name[0].toUpperCase() + name.slice(1)}`);
  if (btn) btn.classList.toggle("active", show);
  if (name === "log" && show) refreshLog();
  if (name === "chart" && show && chart) chart.resize();
}

// ── Init ────────────────────────────────────────────────────
async function init() {
  connectWS();
  await loadParams();
  initLog(ws);
  chart = initChart("chartCanvas");
  if (chart) {
    chart.addSeries("fps", "#0071e3");
    chart.addSeries("detect", "#34c759");
  }
  initROI("roiCanvas");
  initMobileSwipe();
  initTabs();
  setGrid("cols2");

  // Throttle streams periodically
  setInterval(throttleStreams, 5000);

  // Layout buttons
  $("#btnGrid1")?.addEventListener("click", () => setGrid("cols1"));
  $("#btnGrid2")?.addEventListener("click", () => setGrid("cols2"));
  $("#btnGrid4")?.addEventListener("click", () => setGrid("cols4"));
  $("#btnROI")?.addEventListener("click", toggleROI);
  $("#btnSnapshot")?.addEventListener("click", async () => {
    const r = await fetch("/api/snapshot", { method: "POST" });
    const d = await r.json();
    if (d.success) console.log("快照已保存:", d.filename);
  });
  $("#btnRecord")?.addEventListener("click", async function () {
    const recording = this.dataset.recording === "true";
    const r = await fetch("/api/recording", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: recording ? "stop" : "start" }),
    });
    const d = await r.json();
    if (d.success) {
      this.dataset.recording = recording ? "false" : "true";
      this.textContent = recording ? "⏺ 录屏" : "⏹ 停止";
      this.classList.toggle("danger", !recording);
      document.querySelector(".rec-dot")?.classList.toggle("active", !recording);
    }
  });

  // Mobile close buttons
  document.querySelectorAll(".mobile-panel .close-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      btn.parentElement.classList.remove("show");
    });
  });
}

document.addEventListener("DOMContentLoaded", init);
```

- [ ] **Step 2: Write `index.html`**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no" />
  <title>OpenCV 调试面板</title>
  <link rel="stylesheet" href="/static/css/debug.css" />
</head>
<body>

<!-- ═══ Title Bar ═══ -->
<header class="titlebar">
  <div class="dots">
    <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
  </div>
  <span class="title">OpenCV 调试面板</span>
  <span class="ip" id="hostIp"></span>
</header>

<!-- ═══ Main Layout ═══ -->
<div class="main">

  <!-- Sidebar (desktop) -->
  <aside class="sidebar" id="sidebar">
    <!-- Param list populated by params.js -->
    <div id="paramList"></div>

    <div class="sidebar-section">
      <div class="label">布局</div>
      <div class="content">
        <div class="param-row">
          <button id="btnGrid1">1×1</button>
          <button id="btnGrid2" class="primary">1×2</button>
          <button id="btnGrid4">2×2</button>
        </div>
      </div>
    </div>

    <div class="sidebar-section">
      <div class="label">工具</div>
      <div class="content">
        <div class="param-row"><button id="btnROI">📐 ROI 选区</button></div>
        <div class="param-row"><button id="btnSnapshot">📸 快照</button></div>
        <div class="param-row"><button id="btnRecord">⏺ 录屏</button></div>
      </div>
    </div>
  </aside>

  <!-- Viewport -->
  <main class="viewport">
    <div class="grid cols2" id="streamGrid">
      <!-- Populated by layout.js -->
    </div>
  </main>
</div>

<!-- ═══ Bottom Panels (desktop) ═══ -->
<div class="panels">
  <div class="panel-toggle">
    <button id="btnChart">📊 图表</button>
    <button id="btnLog">📋 日志</button>
  </div>
  <div class="panel-chart" id="panelChart">
    <canvas id="chartCanvas"></canvas>
  </div>
  <div class="panel-log" id="panelLog">
    <div id="logList"></div>
  </div>
</div>

<!-- ═══ Status Bar ═══ -->
<footer class="statusbar">
  <div class="metric"><span class="rec-dot" id="recDot"></span></div>
  <div class="metric">◆ FPS: <span class="val" id="metricFps">-</span></div>
  <div class="metric">检测: <span class="val" id="metricDetect">-</span></div>
  <div class="metric">耗时: <span class="val" id="metricLatency">-</span></div>
</footer>

<!-- ═══ Mobile Tab Bar ═══ -->
<nav class="tabbar">
  <button data-panel="view" class="active">
    <span class="icon">📺</span> 视图
  </button>
  <button data-panel="params">
    <span class="icon">🎚</span> 参数
  </button>
  <button data-panel="log">
    <span class="icon">📋</span> 日志
  </button>
</nav>

<!-- Mobile overlays -->
<div class="mobile-panel" id="mobile-params">
  <button class="close-btn">&times;</button>
  <h3>参数调节</h3>
  <div id="mobileParamList"></div>
</div>
<div class="mobile-panel" id="mobile-log">
  <button class="close-btn">&times;</button>
  <h3>日志</h3>
  <div id="mobileLogList"></div>
</div>

<!-- ROI canvas (injected over first stream cell by roi.js) -->
<canvas class="roi-overlay" id="roiCanvas"></canvas>

<script>
  // Show host IP in title bar
  document.getElementById("hostIp").textContent = location.host;
</script>
<script type="module" src="/static/js/app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Commit**

```bash
git add tools/web/static/js/app.js tools/web/templates/index.html
git commit -m "feat(web): add app controller and HTML template"
```

---

### Task 12: Clean up old code

**Files:**
- Modify: `tools/tools.py` — remove `WebStreamer`, `MJPEGHandler`, `detect_cameras`, `camera_init`
- Delete: `tools/shoot.py`

- [ ] **Step 1: Remove web/camera code from `tools/tools.py`**

Remove the following from `tools/tools.py`:
- Lines 1-12: imports of `threading`, `time`, `platform`, `os`, `glob`, `http.server`, `socketserver`, `typing` (keep those still needed)
- Lines 13-210: `MJPEGHandler` class and `WebStreamer` class
- Lines 215-336: `detect_cameras` function (check imports used only by these)
- Lines 339-345: `camera_init` function

Keep: `MV_LABVALUE`, `FpsShow`, `cvt_mvlab2cv`, `order_points`, `perspective_correct_and_validate`, and the `__main__` block.

- [ ] **Step 2: Clean up imports in `tools/tools.py`**

After removing the web and camera code, the remaining imports should be:
```python
import cv2
from cv2.typing import MatLike
import numpy as np
import time
from typing import Optional
```

Remove: `threading`, `platform`, `os`, `glob`, `http.server`, `socketserver`

- [ ] **Step 3: Remove the `__main__` block from `tools/tools.py`**

Remove lines 482-527 (the demo/test block that used `WebStreamer` and `detect_cameras`).

- [ ] **Step 4: Delete `tools/shoot.py`**

```bash
rm tools/shoot.py
```

- [ ] **Step 5: Commit**

```bash
git add tools/tools.py
git rm tools/shoot.py
git commit -m "refactor: remove old web code, migrated to tools/web/"
```

---

### Task 13: Update `main.py` to use new API

**Files:**
- Modify: `main.py`

**Interfaces:**
- Consumes: `DebugServer`, `CameraManager`, `ParamRegistry` from `tools.web`

- [ ] **Step 1: Rewrite `main.py`**

```python
"""去畸变 → 裁剪黑边 → 检测 → 原图画框。"""

import cv2
import numpy as np
from tools.web import DebugServer, CameraManager, ParamRegistry
from tools.tools import FpsShow, cvt_mvlab2cv, perspective_correct_and_validate

# ---- 目标真实宽高比 ----
REAL_ASPECT_RATIO = 0.657
ASPECT_TOLERANCE = 0.4


def main():
    # 1. 打开摄像头
    cam = CameraManager()
    cap = cam.open(index=9, width=1280, height=720, fps=60)

    # 2. 注册可调参数
    params = ParamRegistry()
    params.add("canny_low",  type=int, default=50,  range=(0, 255), group="边缘检测")
    params.add("canny_high", type=int, default=150, range=(0, 255), group="边缘检测")
    params.add("kernel",     type=int, default=5,   range=(1, 21), step=2, group="形态学")
    params.add("min_area",   type=int, default=2000, range=(100, 20000), group="筛选")
    params.add("min_white",  type=int, default=60,  range=(0, 255), group="筛选")

    # 3. 启动 Web 调试面板
    server = DebugServer(params=params, port=8080)
    server.start()

    fps = FpsShow()

    # 相机标定参数
    with np.load("param.npz") as p:
        mtx = p["mtx"]
        dist = p["dist"]

    lab_lower, lab_upper = cvt_mvlab2cv()
    reject_status = {"area": 0, "quad": 0, "white_region": 0, "aspect_ratio": 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 0)

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 读取当前参数
        canny_low = params.get("canny_low")
        canny_high = params.get("canny_high")
        ksize = params.get("kernel")
        min_area = params.get("min_area")
        min_white = params.get("min_white")

        kernel = np.ones((ksize, ksize), np.uint8)
        binary = cv2.inRange(lab, lab_lower, lab_upper)
        blurred = cv2.GaussianBlur(binary, (5, 5), 0)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        reject_status = {"area": 0, "quad": 0, "white_region": 0, "aspect_ratio": 0}

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                reject_status["area"] += 1
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4:
                reject_status["quad"] += 1
                continue

            mask_poly = np.zeros_like(gray)
            cv2.drawContours(mask_poly, [approx], -1, 255, -1)
            white_region = cv2.bitwise_and(gray, gray, mask=mask_poly)
            mean_val = cv2.mean(white_region, mask=mask_poly)[0]
            if mean_val < min_white:
                reject_status["white_region"] += 1
                continue

            pts = approx.reshape(4, 2)
            _, is_valid, real_ratio = perspective_correct_and_validate(
                gray, pts, REAL_ASPECT_RATIO, tolerance=ASPECT_TOLERANCE
            )

            if not is_valid:
                reject_status["aspect_ratio"] += 1
                continue

            for i in range(4):
                cv2.line(frame, approx[i][0], approx[(i + 1) % 4][0], (0, 255, 0), 2)

        fps.show(frame)

        cv2.putText(
            frame,
            f"area:{reject_status['area']} asp:{reject_status['aspect_ratio']} "
            f"quad:{reject_status['quad']} white:{reject_status['white_region']}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # 推流
        server.update_frame(0, frame)
        server.update_frame(1, edges)

        # 日志
        total_reject = sum(reject_status.values())
        if total_reject > 0:
            server.log("REJECT",
                f"area:{reject_status['area']} quad:{reject_status['quad']} "
                f"white:{reject_status['white_region']} asp:{reject_status['aspect_ratio']}",
                "warn" if total_reject > 5 else "info")

        server.metrics.update(
            detect=sum(1 for _ in reject_status.values()),
            latency=0.0,
        )
        server.broadcast_metrics()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "refactor(main): switch to DebugServer API with live param tuning"
```

---

### Task 14: Integration verification

**Files:**
- Verify: all imports resolve, server starts, page loads

- [ ] **Step 1: Verify Python imports**

```bash
cd /home/cmz488/share/tspfile && python -c "from tools.web import DebugServer, CameraManager, ParamRegistry; print('OK')"
```
Expected: prints "OK" without errors.

- [ ] **Step 2: Verify CameraManager.detect()**

```bash
cd /home/cmz488/share/tspfile && python -c "
from tools.web import CameraManager
c = CameraManager()
cams = c.detect()
print(f'Found {len(cams)} cameras')
for cam in cams:
    print(f'  [{cam[\"index\"]}] {cam[\"name\"]} @ {cam[\"default_res\"]}')
"
```
Expected: prints detected cameras or "Found 0 cameras".

- [ ] **Step 3: Verify server starts**

```bash
cd /home/cmz488/share/tspfile && timeout 5 python -c "
from tools.web import DebugServer
s = DebugServer(port=8765)
s.start()
import time; time.sleep(2)
print('Server started successfully')
s.stop()
" 2>&1
```
Expected: prints server URL and "Server started successfully".

- [ ] **Step 4: Verify static files are served**

```bash
cd /home/cmz488/share/tspfile && python -c "
from tools.web import DebugServer
import urllib.request, time, threading

s = DebugServer(port=8766)
s.start()
time.sleep(1)

def check():
    try:
        r = urllib.request.urlopen('http://localhost:8766/static/css/debug.css', timeout=3)
        print(f'CSS: {r.status} (len={len(r.read())})')
        r = urllib.request.urlopen('http://localhost:8766/', timeout=3)
        print(f'HTML: {r.status}')
    except Exception as e:
        print(f'Error: {e}')

t = threading.Thread(target=check); t.start(); t.join(timeout=5)
s.stop()
"
```
Expected: CSS 200, HTML 200.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "verify: integration checks pass for tools/web/"
```
