# 07 — main.py：detect_single() → detect() 多框循环 + 消除重复分支

**What to build:** 两项改进：
1. YOLO 路径从 `detect_single()` 改为 `detect()` 遍历所有检测框，每个 BBox 构建 `best_rect` 传给下游（支持多目标场景）
2. 消除主循环中两处 `yolo_detector is None` 重复分支（检测分支 + 推流分支），提取检测策略为内联函数或策略对象，使新增第三种后端只需改一处

**Blocked by:** 05 — yolo_detector.py 代码气味清理（需要 BBox NamedTuple 稳定接口）。

**Status:** completed

- [ ] YOLO 路径使用 `detector.detect(frame)` 返回列表，遍历每个 BBox
- [ ] 每个 BBox 独立构建 `best_rect`（`np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])`）
- [ ] 检测分支和推流分支的 `yolo_detector is None` 判断合并为一处（如统一 `detection_backend` 变量或策略函数）
- [ ] `USE_YOLO = False` 时 RectTracker 行为不变
- [ ] `USE_YOLO = True` 时多个目标均可检测
