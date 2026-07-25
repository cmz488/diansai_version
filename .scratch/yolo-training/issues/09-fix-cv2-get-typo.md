# 09 — 修复 main.py 预存 bug：cv2.get 拼写错误

**What to build:** 修复 `main.py` 第 112 行预存 bug：`cv2.get(cv2.CAP_PROP_FRAME_WIDGH)` 有两个错误 —— `cv2.get` 不存在（应为 `camera.get`），`CAP_PROP_FRAME_WIDGH` 拼写错误（应为 `CAP_PROP_FRAME_WIDTH`）。同时修 `cv2.CAP_PROP_FRAME_HEIGHT` 的写法一致性（`camera.get(cv2.CAP_PROP_FRAME_HEIGHT)` 无需额外 import）。YOLO 路径下此 bug 每帧触发，需优先修。

**Blocked by:** None — 可立即开始（纯 bugfix，无依赖）。

**Status:** completed

- [ ] `cv2.get(cv2.CAP_PROP_FRAME_WIDGH)` → `camera.get(cv2.CAP_PROP_FRAME_WIDTH)`
- [ ] `cv2.get(cv2.CAP_PROP_FRAME_HEIGHT)` → `camera.get(cv2.CAP_PROP_FRAME_HEIGHT)`
- [ ] YOLO 模式下不再每帧触发 AttributeError
- [ ] RectTracker 模式不受影响（此路径原本 `best_rect` 多为 None，修复后行为一致）
