# Workflow: YOLO 模型训练与部署

## 概述

在 RTX 5060 服务器上用 conda + Ultralytics YOLOv11n 训练目标检测模型，
导出 ONNX → 在 Jetson Orin NX（AllSpark2）上转 TensorRT 引擎 → `YoloDetector`
类提供 ROI 给现有 OpenCV 激光点检测管线。

**触发**：纯手动，每次训练按以下步骤执行。

---

## 前置条件

### 训练端（RTX 5060 Linux 服务器）

| 项目 | 要求 |
|------|------|
| OS | Linux (Ubuntu 22.04+ 或等效) |
| GPU | RTX 5060 (Blackwell, SM 12.x) |
| NVIDIA 驱动 | ≥ 550（支持 CUDA 12.4+） |
| CUDA Toolkit | 12.4 或 12.6（由 conda 环境内的 cudatoolkit 提供亦可） |
| conda | Miniconda3 或 miniforge，未安装则先装 |

### 推理端（Jetson Orin NX / AllSpark2）

| 项目 | 要求 |
|------|------|
| JetPack | ≥ 5.1（自带 TensorRT + trtexec） |
| TensorRT | JetPack 自带，≥ 8.5 |

---

## 目录结构

训练服务器上，在项目根目录下：

```text
diansai_version/
├── yolo/                          # YOLO 训练根目录
│   ├── data.yaml                  # 数据集配置（由 split_dataset.py 生成）
│   ├── train_config.yaml          # 训练参数（手动创建，版本受控）
│   ├── train.py                   # 训练入口脚本
│   ├── raw/                       # 原始标注数据（用户放入）
│   │   ├── images/                #   *.jpg / *.png
│   │   └── labels/                #   *.txt（LabelImg YOLO 格式）
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── runs/                      # Ultralytics 训练产出
│       └── train/
│           └── exp{N}/
│               ├── weights/
│               │   ├── best.pt
│               │   └── last.pt
│               ├── results.csv
│               └── ...
├── scripts/
│   ├── split_dataset.py           # 数据集拆分 + 校验 + 报告
│   └── export_tensorrt.sh         # Jetson 端 ONNX → TensorRT
├── tools/
│   └── yolo_detector.py           # 推理接口类
├── models/                        # Jetson 端推理用模型
│   └── yolov11n_best.engine       # TensorRT 引擎（Jetson 上生成）
└── environment.yml                # conda 环境定义（训练端）
```

---

## 工作流步骤

### 步骤 0：一次性环境搭建（训练服务器）

```bash
# 1. 安装 Miniconda（如未安装）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init

# 2. 从 environment.yml 创建环境
cd diansai_version
conda env create -f environment.yml

# 3. 激活环境
conda activate yolo

# 4. 验证 GPU 可用
python -c "import torch; print(torch.cuda.is_available())"
# 应输出 True
```

### 步骤 1：收集并标注图片

1. 用户通过任意方式收集目标物体图片（相机拍照、录像抽帧等）
2. 将图片放入 `yolo/raw/images/`
3. 用 LabelImg 标注：
   ```bash
   # 安装 LabelImg（一次性）
   pip install labelImg
   # 启动，打开 yolo/raw/images/，标注格式选 YOLO
   labelImg yolo/raw/images/
   ```
4. 标注结果（每张图对应一个同名 `.txt`）自动保存在 `yolo/raw/images/` 下
5. 将 `.txt` 标注文件移动到 `yolo/raw/labels/`（或直接在 LabelImg 里设置保存路径）

### 步骤 2：拆分数据集

```bash
conda activate yolo
python scripts/split_dataset.py --raw yolo/raw --out yolo --ratio 7:2:1
```

脚本行为：
- 读取 `yolo/raw/images/` 和 `yolo/raw/labels/`
- 校验：报告有图无标注 / 有标注无图 / 空标注
- 统计：每个类别的标注框数量
- 随机按 7:2:1 拆分到 `yolo/train/`、`yolo/val/`、`yolo/test/`
- 生成 `yolo/data.yaml`（path, train/val/test, nc, names）
- 输出拆分报告到 stdout

### 步骤 3：训练

```bash
conda activate yolo
python yolo/train.py
```

`yolo/train.py` 读取 `yolo/train_config.yaml`，调用 Ultralytics API：

```yaml
# yolo/train_config.yaml
data: yolo/data.yaml
model: yolov11n.pt          # 自动下载 COCO 预训练权重
epochs: 100
batch: 16                   # RTX 5060 建议 16-32，OOM 则调小
imgsz: 640
```

训练产出在 `yolo/runs/train/exp{N}/`：
- `weights/best.pt` — 验证集上 mAP 最高的权重
- `weights/last.pt` — 最后 epoch 的权重
- `results.csv` — 每 epoch 的 loss/mAP 等指标

### 步骤 4：导出 ONNX（训练服务器）

```bash
conda activate yolo
python -c "
from ultralytics import YOLO
model = YOLO('yolo/runs/train/exp{N}/weights/best.pt')
model.export(format='onnx', imgsz=640, opset=12, simplify=True)
"
# 产出 best.onnx 在同目录下
```

### 步骤 5：转到 TensorRT（Jetson Orin NX）

```bash
# 将 best.onnx scp 到 Jetson 上
scp yolo/runs/train/exp{N}/weights/best.onnx \
    user@jetson:~/diansai_version/models/

# 在 Jetson 上执行
ssh user@jetson
cd ~/diansai_version
bash scripts/export_tensorrt.sh models/best.onnx models/yolov11n_best.engine
```

`scripts/export_tensorrt.sh` 内容：
```bash
#!/bin/bash
# args: <input.onnx> <output.engine>
trtexec \
    --onnx="$1" \
    --saveEngine="$2" \
    --fp16 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x640x640
```

### 步骤 6：集成推理

在 `main.py` 或 `detect_tissue.py` 中用 `YoloDetector` 替换现有矩形检测：

```python
from tools.yolo_detector import YoloDetector

# 初始化（TensorRT 优先，fallback PyTorch）
detector = YoloDetector(
    engine_path="models/yolov11n_best.engine",
    pt_path="models/yolov11n_best.pt",  # 可选 fallback
    conf=0.5,
    imgsz=640,
)

# 检测
for bbox in detector.detect(frame):
    x1, y1, x2, y2, conf, cls_id = bbox
    roi = frame[y1:y2, x1:x2]
    # ... 交给现有 OpenCV 激光点检测
```

---

## 文件清单

以下文件需创建（详见各自章节）：

| 文件 | 用途 | 在哪台机器 |
|------|------|-----------|
| `environment.yml` | conda 环境定义（PyTorch + Ultralytics） | 训练服务器 |
| `scripts/split_dataset.py` | 数据集拆分/校验/统计 | 训练服务器（纯 Python，无 GPU 依赖） |
| `yolo/train_config.yaml` | 训练超参数 | 训练服务器 |
| `yolo/train.py` | 训练入口脚本 | 训练服务器 |
| `scripts/export_tensorrt.sh` | ONNX → TensorRT 转换 | Jetson |
| `tools/yolo_detector.py` | `YoloDetector` 推理类 | Jetson（或两端） |

### `environment.yml`

```yaml
name: yolo
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch>=2.4
  - torchvision
  - pytorch-cuda=12.4
  - pip
  - pip:
      - ultralytics
      - onnx
      - onnxruntime-gpu
      - opencv-python>=5.0
      - labelImg
```

> RTX 5060 (Blackwell) 需 PyTorch ≥ 2.4 + CUDA 12.4。`pytorch-cuda=12.4` 由 conda 的 pytorch channel 提供。

### `scripts/split_dataset.py`

```python
#!/usr/bin/env python3
"""数据集拆分脚本：校验、统计、拆分、生成 data.yaml。"""
import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="YOLO 数据集拆分与校验")
    p.add_argument("--raw", required=True, help="原始数据目录（含 images/ 和 labels/）")
    p.add_argument("--out", required=True, help="输出根目录（生成 train/val/test + data.yaml）")
    p.add_argument("--ratio", default="7:2:1", help="train:val:test 比例")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    return p.parse_args()


def validate(raw: Path):
    """校验图片与标注的对应关系，返回有效配对列表。"""
    img_dir = raw / "images"
    lbl_dir = raw / "labels"
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    img_stems = {f.stem for f in img_dir.glob("*") if f.suffix.lower() in img_exts}
    lbl_stems = {f.stem for f in lbl_dir.glob("*.txt")}

    missing_labels = img_stems - lbl_stems
    missing_images = lbl_stems - img_stems
    empty_labels = []

    pairs = []
    for stem in sorted(img_stems & lbl_stems):
        lbl_path = lbl_dir / f"{stem}.txt"
        content = lbl_path.read_text().strip()
        if not content:
            empty_labels.append(stem)
        else:
            pairs.append(stem)

    # 报告校验结果
    print("=" * 50)
    print("数据集校验报告")
    print("=" * 50)
    print(f"图片总数:   {len(img_stems)}")
    print(f"标注总数:   {len(lbl_stems)}")
    print(f"有效配对:   {len(pairs)}")
    print(f"缺标注:     {len(missing_labels)}", f"({', '.join(sorted(missing_labels)[:5])}...)" if missing_labels else "")
    print(f"缺图片:     {len(missing_images)}", f"({', '.join(sorted(missing_images)[:5])}...)" if missing_images else "")
    print(f"空标注:     {len(empty_labels)}", f"({', '.join(sorted(empty_labels)[:5])}...)" if empty_labels else "")

    # 统计类别
    class_counts = {}
    for stem in pairs:
        for line in (lbl_dir / f"{stem}.txt").read_text().strip().splitlines():
            cls_id = line.split()[0]
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    print(f"\n类别框数统计:")
    for cls_id, count in sorted(class_counts.items()):
        print(f"  class {cls_id}: {count} boxes")

    return pairs, class_counts


def split_and_copy(pairs: list[str], raw: Path, out: Path, ratio: str, seed: int):
    """按比例拆分并复制文件。"""
    r = [int(x) for x in ratio.split(":")]
    total = sum(r)
    train_r, val_r, test_r = r[0] / total, r[1] / total, r[2] / total

    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = round(n * train_r)
    n_val = round(n * val_r)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    img_dir = raw / "images"
    lbl_dir = raw / "labels"

    for split_name, stems in splits.items():
        out_img = out / split_name / "images"
        out_lbl = out / split_name / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            # 找到图片文件（处理多扩展名）
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                src_img = img_dir / f"{stem}{ext}"
                if src_img.exists():
                    shutil.copy2(src_img, out_img / src_img.name)
                    break
            shutil.copy2(lbl_dir / f"{stem}.txt", out_lbl / f"{stem}.txt")

        print(f"  {split_name}: {len(stems)} images")

    # 写 data.yaml
    data_yaml = out / "data.yaml"
    data_yaml.write_text(f"""# Auto-generated by split_dataset.py
path: {out.resolve()}
train: train/images
val: val/images
test: test/images

nc: {len(class_names_hint)}
names: {class_names_hint}
""")
    print(f"\ndata.yaml written to {data_yaml}")


if __name__ == "__main__":
    args = parse_args()
    raw = Path(args.raw)
    out = Path(args.out)

    pairs, class_counts = validate(raw)

    # 尝试从 raw/labels/classes.txt 读取类名，否则用默认
    class_names_hint = {}
    classes_file = raw / "labels" / "classes.txt"
    if classes_file.exists():
        names = classes_file.read_text().strip().splitlines()
        class_names_hint = {i: name for i, name in enumerate(names)}

    split_and_copy(pairs, raw, out, args.ratio, args.seed)
```

### `yolo/train_config.yaml`

```yaml
# YOLO 训练配置
data: yolo/data.yaml        # 数据集配置路径
model: yolov11n.pt          # 预训练权重（自动下载）
epochs: 100                 # 训练轮数
batch: 16                   # batch size（RTX 5060 16G VRAM 建议 16）
imgsz: 640                  # 输入图片尺寸
```

### `yolo/train.py`

```python
#!/usr/bin/env python3
"""YOLO 训练入口，读取 train_config.yaml。"""
import yaml
from pathlib import Path
from ultralytics import YOLO


def main():
    config_path = Path(__file__).parent / "train_config.yaml"
    cfg = yaml.safe_load(config_path.read_text())

    model = YOLO(cfg["model"])
    results = model.train(
        data=cfg["data"],
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        imgsz=cfg["imgsz"],
        # 其余参数使用 Ultralytics 默认值
    )

    print(f"Training complete. Best model at: {model.trainer.best}")
    print(f"Results saved to: {model.trainer.save_dir}")


if __name__ == "__main__":
    main()
```

### `scripts/export_tensorrt.sh`

```bash
#!/bin/bash
# Jetson Orin NX 上执行：ONNX → TensorRT engine
# Usage: bash export_tensorrt.sh <input.onnx> <output.engine>

set -euo pipefail

INPUT="${1:?Usage: $0 <input.onnx> <output.engine>}"
OUTPUT="${2:?}"

echo "Converting ${INPUT} -> ${OUTPUT}"
trtexec \
    --onnx="${INPUT}" \
    --saveEngine="${OUTPUT}" \
    --fp16 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x640x640

echo "Done: ${OUTPUT}"
```

### `tools/yolo_detector.py`

```python
"""YOLO 目标检测推理接口。

Unix 哲学：只负责返回 ROI（bounding box），不关心框内细节。
TensorRT engine 优先加载，fallback 到 PyTorch .pt 文件。
"""

from typing import Optional
import numpy as np

BBox = tuple[int, int, int, int, float, int]
""" (x1, y1, x2, y2, confidence, class_id) — 像素坐标 """


class YoloDetector:
    def __init__(
        self,
        engine_path: Optional[str] = None,
        pt_path: Optional[str] = None,
        conf: float = 0.5,
        imgsz: int = 640,
    ):
        """初始化检测器。

        Args:
            engine_path: TensorRT .engine 文件路径（优先）。
            pt_path: PyTorch .pt 文件路径（fallback）。
            conf: 置信度阈值。
            imgsz: 推理输入尺寸。
        """
        self.conf = conf
        self.imgsz = (imgsz, imgsz)
        self._model = None
        self._backend = "none"

        # 优先 TensorRT
        if engine_path:
            try:
                from ultralytics import YOLO
                self._model = YOLO(engine_path, task="detect")
                self._backend = "tensorrt"
            except Exception as e:
                print(f"[YoloDetector] TensorRT load failed: {e}, trying PyTorch fallback")

        # Fallback: PyTorch
        if self._model is None and pt_path:
            from ultralytics import YOLO
            self._model = YOLO(pt_path, task="detect")
            self._backend = "pytorch"

        if self._model is None:
            raise RuntimeError(
                "YoloDetector: no model loaded. "
                "Provide engine_path or pt_path."
            )

        print(f"[YoloDetector] Backend: {self._backend}, conf={self.conf}")

    def detect(self, frame: np.ndarray) -> list[BBox]:
        """对一帧图像做目标检测。

        Args:
            frame: BGR 图像 (H, W, 3)，OpenCV ndarray。

        Returns:
            BBox 列表，每个元素 (x1, y1, x2, y2, confidence, class_id)。
            按 confidence 降序排列。
        """
        results = self._model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
        boxes_out = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_val = float(box.conf[0])
                cls_id = int(box.cls[0])
                boxes_out.append((x1, y1, x2, y2, conf_val, cls_id))

        # 按置信度降序
        boxes_out.sort(key=lambda b: b[4], reverse=True)
        return boxes_out
```

---

## 与现有代码的集成点

在 `main.py` 或 `detect_tissue.py` 中，现有矩形检测逻辑被替换为：

```python
from tools.yolo_detector import YoloDetector

detector = YoloDetector(
    engine_path="models/yolov11n_best.engine",
    pt_path="models/yolov11n_best.pt",
    conf=0.5,
)

# 在帧循环中
ok, frame = camera.read()
if ok:
    for (x1, y1, x2, y2, conf, cls_id) in detector.detect(frame):
        roi = frame[y1:y2, x1:x2]
        # ROI 交给现有 LaserSpotDetector 做激光点检测
        spot = laser_detector.detect(roi)
```

---

## 检查点（Checkpoint）

本工作流无自动化检查点。以下为人工判断点：

1. **拆分后**：检查 stdout 的校验报告——缺标注的图是否合理？各类别框数是否均衡？
2. **训练后**：检查 `results.csv` 的 mAP50-95 和 loss 曲线——是否收敛？是否需要调参或加数据？
3. **部署前**：在 Jetson 上用几张测试图验证 TensorRT 推理结果和 PyTorch 推理结果一致。
