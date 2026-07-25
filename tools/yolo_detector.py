"""YOLO 目标检测推理接口。

Unix 哲学：只负责返回 ROI（bounding box），不关心框内细节。
TensorRT engine 优先加载，fallback 到 PyTorch .pt 文件。

用法:
    detector = YoloDetector(engine_path="models/best.engine", conf=0.5)
    boxes = detector.detect(frame)  # -> [(x1,y1,x2,y2,conf,cls_id), ...]
"""

from __future__ import annotations

from typing import Optional

import numpy as np

#: BBox 类型: (x1, y1, x2, y2, confidence, class_id) — 像素坐标
BBox = tuple[int, int, int, int, float, int]


class YoloDetector:
    """YOLO 目标检测器 — 输出 ROI 列表，不做框内处理。"""

    def __init__(
        self,
        engine_path: Optional[str] = None,
        pt_path: Optional[str] = None,
        conf: float = 0.5,
        imgsz: int = 640,
    ) -> None:
        """初始化检测器。

        Args:
            engine_path: TensorRT .engine 文件路径（优先加载）。
            pt_path: PyTorch .pt 文件路径（fallback）。
            conf: 置信度阈值，低于此值的检测结果被丢弃。
            imgsz: 推理输入尺寸（正方形，默认 640）。
        """
        self.conf = conf
        self.imgsz = imgsz
        self._model = None
        self._backend: str = "none"

        # 优先 TensorRT engine
        if engine_path:
            self._try_load_engine(engine_path)

        # Fallback: PyTorch .pt
        if self._model is None and pt_path:
            self._try_load_pt(pt_path)

        if self._model is None:
            raise RuntimeError(
                "YoloDetector: no model loaded. "
                "Provide engine_path or pt_path."
            )

        print(f"[YoloDetector] backend={self._backend}, "
              f"conf={self.conf}, imgsz={self.imgsz}")

    # ── 内部加载 ──

    def _try_load_engine(self, path: str) -> None:
        from pathlib import Path
        if not Path(path).exists():
            print(f"[YoloDetector] engine not found: {path}, "
                  f"trying PyTorch fallback")
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(path, task="detect")
            self._backend = "tensorrt"
        except Exception as e:
            print(f"[YoloDetector] TensorRT load failed: {e}")

    def _try_load_pt(self, path: str) -> None:
        from pathlib import Path
        if not Path(path).exists():
            print(f"[YoloDetector] PyTorch model not found: {path}")
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(path, task="detect")
            self._backend = "pytorch"
        except Exception as e:
            print(f"[YoloDetector] PyTorch load failed: {e}")

    # ── 推理 ──

    def detect(self, frame: np.ndarray) -> list[BBox]:
        """对一帧图像做目标检测，返回 ROI 列表。

        Args:
            frame: BGR 图像 (H, W, 3)，OpenCV ndarray。

        Returns:
            BBox 列表，每个元素 (x1, y1, x2, y2, confidence, class_id)。
            按 confidence 降序排列，空列表表示无检测结果。
        """
        if self._model is None:
            return []

        results = self._model(
            frame, conf=self.conf, imgsz=self.imgsz, verbose=False,
        )

        boxes_out: list[BBox] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_val = float(box.conf[0])
                cls_id = int(box.cls[0])
                boxes_out.append((x1, y1, x2, y2, conf_val, cls_id))

        boxes_out.sort(key=lambda b: b[4], reverse=True)
        return boxes_out

    def detect_single(self, frame: np.ndarray) -> Optional[BBox]:
        """返回置信度最高的单个 BBox，无结果时返回 None。"""
        boxes = self.detect(frame)
        return boxes[0] if boxes else None

    @property
    def backend(self) -> str:
        """当前使用的推理后端: 'tensorrt' / 'pytorch' / 'none'。"""
        return self._backend

    @property
    def loaded(self) -> bool:
        """模型是否已成功加载。"""
        return self._model is not None
