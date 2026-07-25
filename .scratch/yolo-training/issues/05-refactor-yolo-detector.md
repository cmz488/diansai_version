# 05 — yolo_detector.py 代码气味清理

**What to build:** 三个纯重构，不改变对外行为：
1. 合并 `_try_load_engine` / `_try_load_pt` 为一个 `_try_load(path, backend_name)` 方法
2. 删除 `loaded` 属性（当前零引用）
3. `BBox` 从裸 `tuple[int,int,int,int,float,int]` 改为 `NamedTuple`，带 `x1/y1/x2/y2/conf/cls_id` 字段名

**Blocked by:** None — 可立即开始。

**Status:** completed

- [ ] `_try_load_engine` 与 `_try_load_pt` 合并为一处，消除重复的 path-check→import→YOLO→set backend 模式
- [ ] `loaded` 属性已删除，`backend` 属性保留
- [ ] `BBox` 使用 `typing.NamedTuple` 定义，字段 `x1, y1, x2, y2, conf, cls_id`
- [ ] `detect()` 和 `detect_single()` 返回类型更新为新 `BBox` 类型
- [ ] 现有调用方（main.py 中的 `x1, y1, x2, y2, conf, cls_id = bbox`）解包不受影响
