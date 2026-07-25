# 02 — 训练脚本 + ONNX 导出

**What to build:** YOLOv11n 训练脚本，通过 `train_config.yaml` 配置驱动（data/epochs/batch/imgsz/model），`python train.py` 零参数启动。训练完成后产出 `best.pt`，附带 ONNX 导出步骤。

**Blocked by:** 01 — conda 环境搭建 + 数据集拆分脚本（需要 conda 环境和 data.yaml）。

**Status:** completed

- [ ] `yolo/train_config.yaml` 存在，含 data/model/epochs/batch/imgsz 字段
- [ ] `yolo/train.py` 零参数启动训练，读取 train_config.yaml
- [ ] 使用 YOLOv11n + COCO 预训练权重，640x640 输入
- [ ] 训练产出在 `yolo/runs/train/exp{N}/` 下，含 `weights/best.pt` 和 `results.csv`
- [ ] `best.pt` 可导出 ONNX（opset=12, simplify=True），`onnx.checker.check_model()` 通过
