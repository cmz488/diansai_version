"""Jetson CSI/USB 摄像头的硬件加速 OpenCV 采集管线。

默认使用 CSI/Argus：

    CSI sensor -> Argus/ISP -> VIC -> OpenCV BGR ndarray

也保留原有 USB MJPEG -> NVDEC -> VIC 链路。检测算法不需要知道
GStreamer 的存在；它只从 :class:`JetsonCamera`
拿到普通的 OpenCV BGR ``ndarray``。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PipelineConfig:
    """摄像头输入和 VIC 转换参数。"""

    source: str = "csi"
    device: str = "/dev/video0"
    sensor_id: int = 0
    sensor_mode: int = -1
    width: int = 1280
    height: int = 720
    fps: int = 60
    flip_method: int = 6
    io_mode: int = 2

    def validate(self) -> None:
        if self.source not in {"csi", "usb"}:
            raise ValueError("source 必须是 'csi' 或 'usb'")
        if self.source == "usb" and not re.fullmatch(r"/dev/video\d+", self.device):
            raise ValueError(f"仅支持 /dev/videoN 设备，实际为 {self.device!r}")
        if self.sensor_id < 0:
            raise ValueError("sensor_id 必须为非负整数")
        if self.sensor_mode < -1:
            raise ValueError("sensor_mode 必须为 -1 或非负整数")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("width、height、fps 必须为正数")
        if self.flip_method not in range(8):
            raise ValueError("flip_method 必须位于 0..7")
        if self.io_mode < 0:
            raise ValueError("io_mode 必须为非负整数")


def build_nvdec_vic_pipeline(config: PipelineConfig) -> str:
    """构造 USB MJPEG -> NVDEC -> VIC -> OpenCV BGR appsink 管线。"""

    config.validate()
    return (
        f"v4l2src device={config.device} io-mode={config.io_mode} ! "
        f"image/jpeg,width={config.width},height={config.height} ! "
        "jpegparse ! "
        "nvv4l2decoder mjpeg=true enable-max-performance=true disable-dpb=true ! "
        f"nvvidconv compute-hw=2 flip-method={config.flip_method} ! "
        "video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def build_argus_vic_pipeline(config: PipelineConfig) -> str:
    """构造 CSI -> Argus/ISP -> VIC -> OpenCV BGR appsink 管线。"""

    config.validate()
    sensor_mode = (
        f" sensor-mode={config.sensor_mode}" if config.sensor_mode >= 0 else ""
    )
    return (
        f"nvarguscamerasrc sensor-id={config.sensor_id}{sensor_mode} ! "
        f"video/x-raw(memory:NVMM),width={config.width},height={config.height},"
        f"format=NV12,framerate={config.fps}/1 ! "
        f"nvvidconv compute-hw=2 flip-method={config.flip_method} ! "
        "video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def build_pipeline(config: PipelineConfig) -> str:
    """按输入类型构造 GStreamer 管线。"""

    if config.source == "csi":
        return build_argus_vic_pipeline(config)
    return build_nvdec_vic_pipeline(config)


class JetsonCamera:
    """把 Jetson GStreamer 管线封装成最小的 OpenCV 风格接口。"""

    def __init__(self, config: PipelineConfig = PipelineConfig()) -> None:
        self.config = config
        self.pipeline: Optional[str] = None
        self._capture: Optional[cv2.VideoCapture] = None
        self._pending_frame: Optional[np.ndarray] = None
        self._last_error: Optional[str] = None

    def open(self) -> bool:
        self.release()
        if self.config.source == "usb" and not os.path.exists(self.config.device):
            self._last_error = f"视频设备不存在：{self.config.device}"
            return False
        try:
            self.pipeline = build_pipeline(self.config)
            capture = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        except Exception as exc:
            self._last_error = str(exc)
            return False

        if not capture.isOpened():
            capture.release()
            if self.config.source == "csi":
                self._last_error = (
                    "无法打开 CSI/Argus 管线；请确认 /dev/video* 已生成、"
                    "nvargus-daemon 正常，并且分辨率/帧率属于传感器支持的模式"
                )
            else:
                self._last_error = (
                    f"无法打开 NVDEC/VIC 管线（设备：{self.config.device}）；"
                    "请确认 USB 相机输出 MJPEG，且 nvv4l2decoder/nvvidconv 可用"
                )
            return False

        # GStreamer/Argus 会延迟到首帧才报告“No cameras available”等错误，
        # 因此 open() 必须实际拉取一帧，不能只相信 isOpened()。
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            if self.config.source == "csi":
                self._last_error = (
                    "CSI/Argus 管线已创建但无法取得首帧；驱动可能没有识别传感器，"
                    "或所选分辨率/帧率不受支持"
                )
            else:
                self._last_error = "USB/NVDEC 管线已创建但无法取得首帧"
            return False

        self._capture = capture
        self._pending_frame = frame
        self._last_error = None
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._capture is None:
            raise RuntimeError("camera 尚未 open")
        if self._pending_frame is not None:
            frame = self._pending_frame
            self._pending_frame = None
            return True, frame
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return False, None
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError("NVDEC/VIC appsink 未返回 BGR HxWx3 ndarray")
        return True, frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._pending_frame = None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def __enter__(self) -> "JetsonCamera":
        if not self.open():
            raise RuntimeError(f"无法打开视频流：{self.last_error}")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


# 兼容项目中旧的 USB 摄像头导入名称。
NvdecVicCamera = JetsonCamera
