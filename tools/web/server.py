"""DebugServer — HTTP + WebSocket orchestration for the OpenCV debug panel."""

import gzip
import io
import json
import os
import time
import struct
import hashlib
import base64
import threading
from email.utils import formatdate, parsedate_to_datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingTCPServer
from typing import Any, Callable, Dict, Optional, Set

from tools.web.streamer import StreamEngine
from tools.web.params import ParamRegistry


# ── WebSocket constants ───────────────────────────────────────

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_WS_OP_TEXT  = 0x1
_WS_OP_CLOSE = 0x8
_WS_OP_PING  = 0x9
_WS_OP_PONG  = 0xA
_GZIP_MIN_BYTES = 256


def _ws_accept_key(key: str) -> str:
    return base64.b64encode(
        hashlib.sha1((key + _WS_GUID).encode()).digest()
    ).decode()


def _ws_send(conn, text: str):
    """Send a WebSocket text frame (server→client, unmasked)."""
    payload = text.encode("utf-8")
    length = len(payload)
    frame = bytearray([0x81])
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
    """Read a WebSocket text frame (client→server, masked)."""
    try:
        b0 = rfile.read(1)
        if not b0: return None
        b1 = rfile.read(1)
        if not b1: return None
        opcode = b0[0] & 0x0F
        if opcode == _WS_OP_CLOSE:
            return None
        if opcode == _WS_OP_PING:
            length = b1[0] & 0x7F
            if length == 126: length = struct.unpack(">H", rfile.read(2))[0]
            elif length == 127: length = struct.unpack(">Q", rfile.read(8))[0]
            if length > 0: rfile.read(length)
            return None
        masked = b1[0] & 0x80
        length = b1[0] & 0x7F
        if length == 126: length = struct.unpack(">H", rfile.read(2))[0]
        elif length == 127: length = struct.unpack(">Q", rfile.read(8))[0]
        if length > 1024 * 1024: return None
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


# ── Static file serving with in-memory cache ──────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE_DIR = os.path.join(_HERE, "templates")
_STATIC_DIR  = os.path.join(_HERE, "static")

_MIME_MAP = {
    ".html": "text/html; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
    ".svg":  "image/svg+xml",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
}

_GZIP_TYPES = frozenset({"text/html", "text/css", "application/javascript",
                          "application/json", "image/svg+xml"})

_static_cache: Dict[str, tuple] = {}
_cache_lock = threading.Lock()


def _cache_static(path: str):
    full = os.path.join(_STATIC_DIR, path.lstrip("/"))
    if not os.path.isfile(full):
        return None
    mtime = os.path.getmtime(full)
    ext = os.path.splitext(full)[1].lower()
    mime_full = _MIME_MAP.get(ext, "application/octet-stream")
    with open(full, "rb") as f:
        raw = f.read()
    base_mime = mime_full.split(";")[0].strip()
    gz_bytes = None
    if base_mime in _GZIP_TYPES and len(raw) >= _GZIP_MIN_BYTES:
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
            gz.write(raw)
        gz_bytes = buf.getvalue()
    entry = (mtime, mime_full, raw, gz_bytes)
    with _cache_lock:
        _static_cache[path] = entry
    return entry


def _get_cached(path: str):
    with _cache_lock:
        entry = _static_cache.get(path)
    full = os.path.join(_STATIC_DIR, path.lstrip("/"))
    if not os.path.isfile(full):
        return None
    current_mtime = os.path.getmtime(full)
    if entry is None or entry[0] != current_mtime:
        entry = _cache_static(path)
    return entry


def _serve_static(path: str, handler) -> bool:
    if ".." in path:
        handler.send_error(403)
        return True
    entry = _get_cached(path)
    if entry is None:
        return False
    mtime, mime_full, raw, gz_bytes = entry
    last_modified = formatdate(mtime, usegmt=True)

    if_modified = handler.headers.get("If-Modified-Since")
    if if_modified:
        try:
            since = parsedate_to_datetime(if_modified).timestamp()
            if since >= mtime:
                handler.send_response(304)
                handler.end_headers()
                return True
        except (ValueError, TypeError):
            pass

    accept_encoding = handler.headers.get("Accept-Encoding", "")
    use_gzip = gz_bytes is not None and "gzip" in accept_encoding
    body = gz_bytes if use_gzip else raw

    handler.send_response(200)
    handler.send_header("Content-Type", mime_full)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Last-Modified", last_modified)
    handler.send_header("Cache-Control", "public, max-age=300")
    if use_gzip:
        handler.send_header("Content-Encoding", "gzip")
    handler.end_headers()
    handler.wfile.write(body)
    return True


# ── HTTP request handler ──────────────────────────────────────

class _DebugHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        ds: "DebugServer" = self.server.debug_server

        if self.path == "/":
            return self._serve_template()
        if self.path.startswith("/static/"):
            path = self.path[len("/static/"):]
            if _serve_static(path, self):
                return
            self.send_error(404)
            return
        if self.path.startswith("/stream/"):
            return self._serve_stream()
        if self.path == "/ws":
            return self._ws_handler()
        if self.path == "/api/params":
            return self._json(ds.params.list_all())
        if self.path == "/api/metrics":
            return self._json(ds.metrics.snapshot())
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
        self.send_error(404)

    # ── helpers ───────────────────────────────────────────────

    def _serve_template(self):
        # template is cached under _static_cache["index.html"] during _warm_cache
        with _cache_lock:
            entry = _static_cache.get("index.html")
        if entry is None:
            html = "<html><body><h1>模板未找到</h1></body></html>".encode("utf-8")
            mime = "text/html; charset=utf-8"
            gz_bytes = None
        else:
            _, mime, html, gz_bytes = entry
        accept = self.headers.get("Accept-Encoding", "")
        use_gzip = gz_bytes is not None and "gzip" in accept
        body = gz_bytes if use_gzip else html
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self):
        try:
            ch = int(self.path.split("/")[-1])
        except ValueError:
            self.send_error(400)
            return
        ds: "DebugServer" = self.server.debug_server
        ds.streamer.subscribe(ch)
        boundary = b"frame"
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                jpeg = ds.streamer.get_jpeg(ch)
                if jpeg is not None:
                    header = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    )
                    self.wfile.write(header + jpeg + b"\r\n")
                else:
                    time.sleep(0.03)
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            ds.streamer.unsubscribe(ch)

    def _ws_handler(self):
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
        accept = self.headers.get("Accept-Encoding", "")
        use_gzip = len(body) >= _GZIP_MIN_BYTES and "gzip" in accept
        if use_gzip:
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
                gz.write(body)
            body = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
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
    ):
        self.params = params or ParamRegistry()
        self.streamer = StreamEngine(max_channels=8)
        self.metrics = MetricsCollector()

        self._port = port
        self._host = host
        self._server: Optional[ThreadingTCPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._ws_clients: Set[Any] = set()
        self._ws_lock = threading.Lock()

        self._metrics_interval = 1.0
        self._metrics_last = 0.0

        self._warm_cache()

    # ── lifecycle ─────────────────────────────────────────────

    def _warm_cache(self):
        for root, _, files in os.walk(_STATIC_DIR):
            for fn in files:
                rel = os.path.relpath(os.path.join(root, fn), _STATIC_DIR)
                _cache_static(rel)
        tp = os.path.join(_TEMPLATE_DIR, "index.html")
        if os.path.isfile(tp):
            with _cache_lock:
                with open(tp, "rb") as f:
                    raw = f.read()
                buf = io.BytesIO()
                with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
                    gz.write(raw)
                _static_cache["index.html"] = (
                    os.path.getmtime(tp),
                    "text/html; charset=utf-8",
                    raw,
                    buf.getvalue(),
                )

    def start(self):
        if self._running:
            return

        class _Server(ThreadingTCPServer, HTTPServer):
            allow_reuse_address = True

        self._server = _Server((self._host, self._port), _DebugHandler)
        self._server.debug_server = self
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._running = True
        print(f"\n[DebugServer] http://{self._host}:{self._port}")
        print(f"[DebugServer] WebSocket ws://{self._host}:{self._port}/ws\n")

    def stop(self):
        self._running = False
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        print("[DebugServer] 已关闭")

    # ── public API ────────────────────────────────────────────

    def update_frame(self, channel_id: int, frame):
        self.streamer.update(channel_id, frame)

    # ── WebSocket internals ───────────────────────────────────

    def _ws_add(self, conn):
        with self._ws_lock:
            self._ws_clients.add(conn)
        self._ws_send(conn, {
            "type": "params",
            "params": self.params.snapshot(),
        })
        self._ws_send(conn, {
            "type": "conn",
            "count": len(self._ws_clients),
        })

    def _ws_remove(self, conn):
        with self._ws_lock:
            self._ws_clients.discard(conn)

    def _ws_dispatch(self, raw: str):
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

    def _broadcast_params(self):
        self._broadcast_raw(json.dumps({
            "type": "params",
            "params": self.params.snapshot(),
        }, ensure_ascii=False))

    def _ws_send(self, conn, obj: dict):
        try:
            _ws_send(conn, json.dumps(obj, ensure_ascii=False))
        except Exception:
            self._ws_remove(conn)

    def _broadcast_raw(self, text: str):
        dead = set()
        with self._ws_lock:
            for conn in self._ws_clients:
                try:
                    _ws_send(conn, text)
                except Exception:
                    dead.add(conn)
            self._ws_clients -= dead

    def broadcast_metrics(self):
        now = time.time()
        if now - self._metrics_last < self._metrics_interval:
            return
        self._metrics_last = now
        self._broadcast_raw(json.dumps({
            "type": "metrics",
            **self.metrics.snapshot(),
        }, ensure_ascii=False))
