# 08 — 接通 ROI→激光点检测管线

**What to build:** 取消 `main.py` 中被注释的 `LaserSpotDetector` 初始化与推理代码。YOLO 检测的每个 ROI 传给 `laser_detector.detect(frame, search_polygon=best_rect)`，Web 调试界面显示激光点质心位置（十字 + 置信度标签）。

**Blocked by:** 07 — detect() 多框循环（需要每个 BBox 独立的 best_rect 才能正确传递 ROI 给激光检测）。

**Status:** completed

- [ ] `LaserSpotDetector` 初始化代码（第 69-83 行）取消注释
- [ ] 帧循环中 YOLO 路径下，每个 `best_rect` 传入 `laser_detector.detect(frame, search_polygon=best_rect)`
- [ ] 激光点检测结果通过 `graph.draw_point/draw_cross/draw_label` 显示在 Web 调试界面
- [ ] `USE_YOLO = False` 时 RectTracker 路径也接入激光检测（与原注释行为一致）
- [ ] Web 界面正常显示：YOLO 框 + 框内激光点
