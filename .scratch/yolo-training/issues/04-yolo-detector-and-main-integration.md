# 04 — YoloDetector 推理接口 + main.py 集成

**What to build:** `YoloDetector` 类封装 YOLO 推理，TensorRT engine 优先加载，PyTorch .pt fallback。`detect(frame) -> list[(x1,y1,x2,y2,conf,cls_id)]` 返回像素坐标 ROI 列表。修改 `main.py`：YOLO 检测取代当前 RectTracker 矩形定位 → ROI 交给现有 LaserSpotDetector 做激光点质心计算。

**Blocked by:** 03 — Jetson 端 TensorRT 引擎转换（需要 .engine 做集成验证；开发阶段可用 .pt fallback 并行推进）。

**Status:** completed

- [ ] `tools/yolo_detector.py` 存在，导出 `YoloDetector` 类
- [ ] 构造函数接受 `engine_path`（优先）和 `pt_path`（fallback），`conf` 阈值默认 0.5
- [ ] `detect(frame: np.ndarray) -> list[tuple[int,int,int,int,float,int]]` 返回像素坐标 BBox，按置信度降序
- [ ] TensorRT engine 路径有效时优先加载；无效则 fallback 到 PyTorch .pt
- [ ] `main.py` 集成：帧循环中调用 `detector.detect(frame)` 获取 ROI，替代当前 RectTracker 矩形检测逻辑
- [ ] YOLO ROI 传递给现有 LaserSpotDetector，Web 调试界面正常显示检测结果
- [ ] PyTorch fallback 路径验证：删除 engine 文件后能正常降级运行
