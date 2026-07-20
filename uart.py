import time
import cv2
import numpy as np
from tools.peripheral import Uart
from tools.hardware_pipeline import PipelineConfig, JetsonCamera

if __name__ == "__main__":
    config = PipelineConfig(
        source="usb",
        device="/dev/video0",
        width=640,
        height=480,
        fps=60,
        flip_method=6,
    )
    camera = JetsonCamera(config)
    camera.open()
    ret, frame = camera.read()
    cv2.imwrite("1.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
