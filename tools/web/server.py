"""DebugServer — HTTP + WebSocket orchestration for the OpenCV debug panel."""

import json
import os
import time
import struct
import hashlib
import base64
import threading
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
            length = b1[0] & 0x7F
            if length == 126:
                length = struct.unpack(">H", rfile.read(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", rfile.read(8))[0]
            if length > 0:
                rfile.read(length)
            return None

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
            path = self.path[len("/static/"):]
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
    """Top-level orchestrator for the OpenCV web debug panel."""

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

        self._ws_clients: Set[Any] = set()
        self._ws_lock = threading.Lock()

        self._camera_list_cache: List[Dict] = []

        self._metrics_interval = 1.0
        self._metrics_last = 0.0

        self.log_buffer.on_append(self._on_log_entry)

    # ── lifecycle ─────────────────────────────────────────────

    def start(self):
        """Start HTTP + WS server in a background daemon thread."""
        if self._running:
            return

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
        if self.recorder.is_recording and channel_id == 0:
            self.recorder.add_frame(frame)

    def log(self, tag: str, msg: str, level: str = "info"):
        """Append a structured log entry (broadcast to WS clients)."""
        return self.log_buffer.append(tag, msg, level)

    # ── WebSocket internals ───────────────────────────────────

    def _ws_add(self, conn):
        with self._ws_lock:
            self._ws_clients.add(conn)
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
            pass

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
