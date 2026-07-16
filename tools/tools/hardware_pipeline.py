"""Jetson hardware acceleration exposed through OpenCV-compatible interfaces.

The capture side keeps MJPEG decode and image transform on NVDEC/VIC, then returns
ordinary BGR numpy.ndarray frames from read(). The compute side exposes the same
tuple as tools.tools.preprocess so existing OpenCV detection code does not need
to know which backend produced the images.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


CAPTURE_BACKENDS = ("auto", "nvdec_vic", "v4l2")
COMPUTE_BACKENDS = ("auto", "cpu", "umat", "vpi_cuda")


@dataclass(frozen=True)
class PipelineConfig:
    device: str = "/dev/video0"
    width: int = 1280
    height: int = 720
    fps: int = 60
    flip_method: int = 6
    io_mode: int = 2
    fourcc: str = "MJPG"

    def validate(self) -> None:
        if not re.fullmatch(r"/dev/video\d+", self.device):
            raise ValueError(f"仅支持 /dev/videoN 设备，实际为 {self.device!r}")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("width、height、fps 必须为正数")
        if self.flip_method not in range(8):
            raise ValueError("flip_method 必须位于 0..7")
        if len(self.fourcc) != 4:
            raise ValueError("fourcc 必须是 4 个字符")


@dataclass(frozen=True)
class HardwareCapabilities:
    opencv_version: str
    gstreamer: bool
    nvdec: bool
    vic: bool
    opencl: bool
    opencl_enabled: bool
    opencv_cuda_devices: int
    vpi: bool
    vpi_version: Optional[str]
    pycuda: bool
    pycuda_devices: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _opencv_has_gstreamer() -> bool:
    info = cv2.getBuildInformation()
    return bool(re.search(r"GStreamer:\s+YES", info))


def _has_gst_element(name: str) -> bool:
    if shutil.which("gst-inspect-1.0") is None:
        return False
    result = subprocess.run(
        ["gst-inspect-1.0", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


@lru_cache(maxsize=1)
def probe_capabilities() -> HardwareCapabilities:
    """Probe real runtime support instead of assuming Jetson features are usable."""
    opencl = bool(cv2.ocl.haveOpenCL())
    if opencl:
        cv2.ocl.setUseOpenCL(True)
    opencl_enabled = bool(cv2.ocl.useOpenCL())

    try:
        opencv_cuda_devices = int(cv2.cuda.getCudaEnabledDeviceCount())
    except (AttributeError, cv2.error):
        opencv_cuda_devices = 0

    vpi_available = False
    vpi_version: Optional[str] = None
    try:
        import vpi  # type: ignore

        vpi_available = True
        vpi_version = str(getattr(vpi, "__version__", "unknown"))
    except Exception:
        pass

    pycuda_available = False
    pycuda_devices = 0
    try:
        import pycuda.driver as cuda  # type: ignore

        cuda.init()
        pycuda_devices = int(cuda.Device.count())
        pycuda_available = pycuda_devices > 0
    except Exception:
        pass

    gstreamer = _opencv_has_gstreamer()
    return HardwareCapabilities(
        opencv_version=cv2.__version__,
        gstreamer=gstreamer,
        nvdec=gstreamer and _has_gst_element("nvv4l2decoder"),
        vic=gstreamer and _has_gst_element("nvvidconv"),
        opencl=opencl,
        opencl_enabled=opencl_enabled,
        opencv_cuda_devices=opencv_cuda_devices,
        vpi=vpi_available,
        vpi_version=vpi_version,
        pycuda=pycuda_available,
        pycuda_devices=pycuda_devices,
    )


def build_nvdec_vic_pipeline(config: PipelineConfig) -> str:
    """Build USB MJPEG -> NVDEC -> VIC -> OpenCV BGR appsink pipeline."""
    config.validate()
    return (
        f"v4l2src device={config.device} io-mode={config.io_mode} ! "
        f"image/jpeg,width={config.width},height={config.height},"
        f"framerate={config.fps}/1 ! "
        "jpegparse ! "
        "nvv4l2decoder mjpeg=true enable-max-performance=true disable-dpb=true ! "
        f"nvvidconv compute-hw=2 flip-method={config.flip_method} ! "
        "video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


class OpenCVHardwareCapture:
    """Small cv2.VideoCapture-style wrapper with safe hardware fallback."""

    def __init__(
        self,
        config: PipelineConfig = PipelineConfig(),
        backend: str = "auto",
    ) -> None:
        if backend not in CAPTURE_BACKENDS:
            raise ValueError(f"capture backend 必须是 {CAPTURE_BACKENDS}")
        self.config = config
        self.requested_backend = backend
        self.active_backend: Optional[str] = None
        self.pipeline: Optional[str] = None
        self._capture: Optional[cv2.VideoCapture] = None
        self._last_error: Optional[str] = None

    def open(self) -> bool:
        self.release()
        self.config.validate()
        candidates = (
            ("nvdec_vic", "v4l2")
            if self.requested_backend == "auto"
            else (self.requested_backend,)
        )
        errors = []
        for backend in candidates:
            try:
                capture = self._open_backend(backend)
            except Exception as exc:
                errors.append(f"{backend}: {exc}")
                continue
            if capture.isOpened():
                self._capture = capture
                self.active_backend = backend
                self._last_error = None
                return True
            capture.release()
            errors.append(f"{backend}: open failed")

        self._last_error = "; ".join(errors)
        return False

    def _open_backend(self, backend: str) -> cv2.VideoCapture:
        if backend == "nvdec_vic":
            caps = probe_capabilities()
            if not (caps.gstreamer and caps.nvdec and caps.vic):
                raise RuntimeError("OpenCV GStreamer 或 NVDEC/VIC 插件不可用")
            self.pipeline = build_nvdec_vic_pipeline(self.config)
            return cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)

        if backend == "v4l2":
            self.pipeline = None
            index = int(self.config.device.rsplit("video", 1)[1])
            capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if capture.isOpened():
                capture.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter.fourcc(*self.config.fourcc),
                )
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                capture.set(cv2.CAP_PROP_FPS, self.config.fps)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture

        raise ValueError(f"未知 capture backend: {backend}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._capture is None:
            raise RuntimeError("capture 尚未 open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return False, None
        if self.active_backend == "v4l2":
            frame = self._software_flip(frame, self.config.flip_method)
        return True, frame

    @staticmethod
    def _software_flip(frame: np.ndarray, method: int) -> np.ndarray:
        if method == 0:
            return frame
        if method == 1:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if method == 2:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if method == 3:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if method == 4:
            return cv2.flip(frame, 1)
        if method == 5:
            return cv2.flip(cv2.transpose(frame), -1)
        if method == 6:
            return cv2.flip(frame, 0)
        if method == 7:
            return cv2.transpose(frame)
        raise ValueError(f"未知 flip_method: {method}")

    def isOpened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def get(self, prop_id: int) -> float:
        return 0.0 if self._capture is None else float(self._capture.get(prop_id))

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self.active_backend = None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def __enter__(self) -> "OpenCVHardwareCapture":
        if not self.open():
            raise RuntimeError(f"无法打开视频流：{self.last_error}")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


class OpenCVComputePipeline:
    """OpenCV-compatible preprocessing with CPU, UMat or VPI CUDA backends."""

    def __init__(self, backend: str = "auto") -> None:
        if backend not in COMPUTE_BACKENDS:
            raise ValueError(f"compute backend 必须是 {COMPUTE_BACKENDS}")
        self.requested_backend = backend
        self.active_backend = self._select_backend(backend)
        self._vpi_state: Optional[Dict[str, object]] = None

    @staticmethod
    def _select_backend(backend: str) -> str:
        if backend == "auto":
            # End-to-end benchmarks on this project are camera-limited and the
            # CPU chain is marginally faster/more stable than VPI after copies.
            return "cpu"
        if backend == "umat":
            cv2.ocl.setUseOpenCL(True)
            if not cv2.ocl.useOpenCL():
                raise RuntimeError(
                    "OpenCV UMat 可调用，但当前没有 OpenCL 设备；"
                    "Jetson NVIDIA GPU 应使用 VPI/CUDA，而不是伪装成 UMat 加速"
                )
        if backend == "vpi_cuda":
            try:
                import vpi  # noqa: F401
            except Exception as exc:
                raise RuntimeError(f"VPI Python 不可用: {exc}") from exc
        return backend

    def preprocess(
        self,
        frame: np.ndarray,
        kernel: np.ndarray,
        rect_lab_thresholds: Tuple[Sequence[int], Sequence[int]],
        laser_mask_thresholds: Tuple[Sequence[int], Sequence[int]],
        canny_thresholds: Tuple[int, int] = (50, 150),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame 必须是 HxWx3 BGR ndarray")
        if kernel.ndim != 2:
            raise ValueError("kernel 必须是二维 ndarray")
        height, width = kernel.shape
        if height <= 0 or width <= 0 or height % 2 == 0 or width % 2 == 0:
            raise ValueError("kernel 宽高必须为正奇数")

        if self.active_backend == "cpu":
            return self._preprocess_cpu(
                frame,
                kernel,
                rect_lab_thresholds,
                laser_mask_thresholds,
                canny_thresholds,
            )
        if self.active_backend == "umat":
            return self._preprocess_umat(
                frame,
                kernel,
                rect_lab_thresholds,
                laser_mask_thresholds,
                canny_thresholds,
            )
        if self.active_backend == "vpi_cuda":
            try:
                return self._preprocess_vpi_cuda(
                    frame,
                    kernel,
                    rect_lab_thresholds,
                    laser_mask_thresholds,
                    canny_thresholds,
                )
            except Exception as exc:
                if self.requested_backend != "auto":
                    raise
                warnings.warn(
                    f"VPI CUDA 初始化失败，自动回退 CPU: {exc}",
                    RuntimeWarning,
                )
                self.active_backend = "cpu"
                self._vpi_state = None
                return self._preprocess_cpu(
                    frame,
                    kernel,
                    rect_lab_thresholds,
                    laser_mask_thresholds,
                    canny_thresholds,
                )
        raise AssertionError(self.active_backend)

    @staticmethod
    def _preprocess_cpu(
        frame: np.ndarray,
        kernel: np.ndarray,
        rect_thresholds: Tuple[Sequence[int], Sequence[int]],
        laser_thresholds: Tuple[Sequence[int], Sequence[int]],
        canny_thresholds: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rect_binary = cv2.inRange(lab, rect_thresholds[0], rect_thresholds[1])
        blurred = cv2.GaussianBlur(rect_binary, kernel.shape[::-1], 0)
        edges = cv2.Canny(blurred, canny_thresholds[0], canny_thresholds[1])
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        laser = cv2.inRange(lab, laser_thresholds[0], laser_thresholds[1])
        return edges, laser, gray

    @staticmethod
    def _preprocess_umat(
        frame: np.ndarray,
        kernel: np.ndarray,
        rect_thresholds: Tuple[Sequence[int], Sequence[int]],
        laser_thresholds: Tuple[Sequence[int], Sequence[int]],
        canny_thresholds: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = cv2.UMat(frame)
        lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        rect_binary = cv2.inRange(lab, rect_thresholds[0], rect_thresholds[1])
        blurred = cv2.GaussianBlur(rect_binary, kernel.shape[::-1], 0)
        edges = cv2.Canny(blurred, canny_thresholds[0], canny_thresholds[1])
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        laser = cv2.inRange(lab, laser_thresholds[0], laser_thresholds[1])
        return edges.get(), laser.get(), gray.get()

    def _preprocess_vpi_cuda(
        self,
        frame: np.ndarray,
        kernel: np.ndarray,
        rect_thresholds: Tuple[Sequence[int], Sequence[int]],
        laser_thresholds: Tuple[Sequence[int], Sequence[int]],
        canny_thresholds: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import vpi  # type: ignore

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laser = cv2.inRange(lab, laser_thresholds[0], laser_thresholds[1])

        shape = frame.shape[:2]
        kernel_shape = tuple(int(value) for value in kernel.shape)
        state = self._vpi_state
        if (
            state is None
            or state["shape"] != shape
            or state["kernel_shape"] != kernel_shape
        ):
            rect_binary = np.empty(shape, dtype=np.uint8)
            image = vpi.asimage(rect_binary)
            sigma = 0.3 * ((kernel_shape[1] - 1) * 0.5 - 1) + 0.8
            gaussian = image.gaussian_filter(
                kernel_shape[1],
                sigma,
                backend=vpi.Backend.CUDA,
                border=vpi.Border.CLAMP,
            )
            edges = gaussian.canny(
                thresh_weak=canny_thresholds[0],
                thresh_strong=canny_thresholds[1],
                edge_value=255,
                nonedge_value=0,
                backend=vpi.Backend.CUDA,
            )
            dilated = edges.dilate(
                kernel_shape,
                backend=vpi.Backend.CUDA,
                border=vpi.Border.CLAMP,
            )
            output = dilated.erode(
                kernel_shape,
                backend=vpi.Backend.CUDA,
                border=vpi.Border.CLAMP,
            )
            state = {
                "shape": shape,
                "kernel_shape": kernel_shape,
                "rect_binary": rect_binary,
                "image": image,
                "gaussian": gaussian,
                "edges": edges,
                "dilated": dilated,
                "output": output,
                "sigma": sigma,
            }
            self._vpi_state = state

        rect_binary = state["rect_binary"]
        cv2.inRange(
            lab,
            rect_thresholds[0],
            rect_thresholds[1],
            dst=rect_binary,
        )
        image = state["image"]
        gaussian = state["gaussian"]
        edges = state["edges"]
        dilated = state["dilated"]
        output = state["output"]
        sigma = state["sigma"]
        image.gaussian_filter(
            kernel_shape[1],
            sigma,
            backend=vpi.Backend.CUDA,
            border=vpi.Border.CLAMP,
            out=gaussian,
        )
        gaussian.canny(
            thresh_weak=canny_thresholds[0],
            thresh_strong=canny_thresholds[1],
            edge_value=255,
            nonedge_value=0,
            backend=vpi.Backend.CUDA,
            out=edges,
        )
        edges.dilate(
            kernel_shape,
            backend=vpi.Backend.CUDA,
            border=vpi.Border.CLAMP,
            out=dilated,
        )
        dilated.erode(
            kernel_shape,
            backend=vpi.Backend.CUDA,
            border=vpi.Border.CLAMP,
            out=output,
        )
        return np.asarray(output.cpu()), laser, gray


class OpenCVHardwarePipeline:
    """Combined capture and preprocessing facade used by the application."""

    def __init__(
        self,
        config: PipelineConfig = PipelineConfig(),
        capture_backend: str = "auto",
        compute_backend: str = "auto",
    ) -> None:
        self.capture = OpenCVHardwareCapture(config, capture_backend)
        self.compute = OpenCVComputePipeline(compute_backend)

    def open(self) -> bool:
        return self.capture.open()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.capture.read()

    def preprocess(
        self,
        frame: np.ndarray,
        kernel: np.ndarray,
        rect_lab_thresholds: Tuple[Sequence[int], Sequence[int]],
        laser_mask_thresholds: Tuple[Sequence[int], Sequence[int]],
        canny_thresholds: Tuple[int, int] = (50, 150),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.compute.preprocess(
            frame,
            kernel,
            rect_lab_thresholds,
            laser_mask_thresholds,
            canny_thresholds,
        )

    @property
    def capture_backend(self) -> Optional[str]:
        return self.capture.active_backend

    @property
    def compute_backend(self) -> str:
        return self.compute.active_backend

    def release(self) -> None:
        self.capture.release()

    def __enter__(self) -> "OpenCVHardwarePipeline":
        if not self.open():
            raise RuntimeError(f"无法打开视频流：{self.capture.last_error}")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
