# NOTES.md — 用户世界

## 硬件拓扑
- **训练端**：RTX 5060（Blackwell 架构，需 CUDA 12.4+），Linux，conda + PyTorch 未安装
- **推理端**：Jetson Orin NX（项目代号 AllSpark2，CSI 摄像头 + OpenCV + TensorRT）
- 模式：服务器训练 → 导出 ONNX → Jetson 端 trtexec 转 TensorRT 引擎 → 推理

## 模型选型
- 框架：Ultralytics YOLOv11
- 模型尺寸：nano（YOLOv11n），COCO 预训练权重
- 输入尺寸：640x640，epochs 默认 100
- 导出链路：PyTorch → ONNX → Jetson 端 trtexec → TensorRT 引擎

## 当前项目
- `diansai_version`：基于 OpenCV 的传统 CV 检测（LAB 阈值、激光点检测）
- 包管理：pip/uv（pyproject.toml），暂无 conda
- YOLO 定位：负责检测目标物体并返回 ROI（bounding box）
- OpenCV 细检：在 ROI 内做亚像素级激光点质心计算（保留现有 LAB 阈值等参数）
- 架构原则：Unix 哲学——YOLO 只管 ROI，OpenCV 只管 ROI 内精检，两者解耦

## 工作流交付物（已细化 → workflows/yolo-training.md）
1. **标注脚本** — `scripts/split_dataset.py`：拆分 + 校验 + 统计 + 生成 data.yaml
2. **YOLO 训练框架搭建** — `environment.yml`（conda env `yolo`, Python 3.10, PyTorch 2.4 + CUDA 12.4）
3. **YOLO 训练脚本** — `yolo/train.py` + `yolo/train_config.yaml`
4. **YOLO 模型使用接口** — `tools/yolo_detector.py`：`YoloDetector` 类
5. **导出脚本** — `scripts/export_tensorrt.sh`（Jetson 端 trtexec）

## 数据集约定
- 图片 + 标注 + class.txt 放在 `yolo/` 目录下
- 标注格式：YOLO txt（LabelImg 输出）
- 拆分比例：train:val:test = 7:2:1
- 标注辅助脚本功能：拆分三目录 + 校验缺漏（有图无标注/有标注无图）+ 类别框数统计报告 + 生成 data.yaml
- 图片由用户自行收集，不需要 Jetson 端采集脚本
- 训练产出：`yolo/runs/train/exp{N}/`（Ultralytics 原生目录），`best.pt` 导出 ONNX → Jetson 端 `models/*.engine`

## 环境与触发
- conda 环境名：`yolo`，Python 3.10
- 训练服务器未安装 conda/CUDA/PyTorch，需从零配
- 触发方式：纯手动——收集图片 → 标注 → 拆分脚本 → 训练 → 导出 → 部署
## 推理接口约定
- BBox 格式：`(x1, y1, x2, y2, confidence, class_id)`，像素坐标
- conf 阈值：默认 0.5，`YoloDetector(conf=0.5)` 可覆盖
- TensorRT 转换：JetPack 自带 `trtexec` CLI
- 训练配置：`train_config.yaml` 仅含 data/epochs/batch/imgsz/model，其余用 Ultralytics 默认
