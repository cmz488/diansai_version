# OpenCV Web Debugger — 工程设计文档

## 概述

将 `tools/tools.py` 和 `tools/shoot.py` 中的 web 功能统一为一个工程化的 OpenCV 调试和展示 web 工具。目标平台：泰山派 ARM 板（RK3588 级别），需同时适配桌面和手机。

## 设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 架构 | 模块化后端 + 组件化前端 | 职责单一、可测试，不过度抽象 |
| 前端 | 模板分离，零依赖原生 JS | 泰山派性能受限，不引入框架 |
| 会话模型 | 服务端参数共享 | 一人调参全员可见，匹配团队调试场景 |
| 旧代码 | 不兼容，直接删除 | 干净重构，不拖旧包袱 |
| 风格 | Apple 白色风格 + 响应式 | 用户喜好 |

## 文件结构

```
tools/web/
├── __init__.py              # from tools.web import DebugServer
├── server.py                # DebugServer — 生命周期编排、路由、WS
├── streamer.py              # StreamEngine — N通道 MJPEG 引擎
├── camera.py                # CameraManager — 整合摄像头探测 + 初始化
├── params.py                # ParamRegistry — 参数注册/校验/回调
├── recorder.py              # Recorder — 快照 + 录屏
├── log_buffer.py            # LogBuffer — 环形日志缓冲
├── templates/
│   └── index.html           # 页面骨架
└── static/
    ├── css/
    │   └── debug.css        # 苹果风 + 响应式
    └── js/
        ├── app.js           # 主控: WS 连接、Tab 切换、路由
        ├── params.js        # 参数面板组件
        ├── layout.js        # 视图网格 + 响应式
        ├── roi.js           # ROI 选区 overlay
        ├── chart.js         # Canvas 实时折线图
        ├── log.js           # 虚拟滚动日志
        └── utils.js         # 防抖、DOM 工具
```

## 后端模块

### server.py — DebugServer

路由分发、WebSocket 升级、启动/停止生命周期。持有所有子模块的引用。

**公开 API：**
- `start()` — 非阻塞，在后台 daemon 线程启动 HTTP 服务器
- `stop()` — 安全关闭
- `update_frame(channel_id, frame)` — 推送帧到指定通道（委托给 StreamEngine）
- `log(tag, msg, level="info")` — 写入结构化日志（委托给 LogBuffer）
- `metrics` — MetricsCollector 实例，`.update(**kv)` 更新指标

**路由表：**

```
GET  /                    → 渲染 index.html
GET  /static/*            → 静态文件服务 (css/, js/)

── 视频流 ──
GET  /stream/<channel_id> → MJPEG 流，query: ?quality=70&maxfps=15

── 参数 ──
GET  /api/params          → 列出所有参数 [{name, type, value, min, max, step, group}]
POST /api/params          → 更新参数 {name: "threshold", value: 127}

── 摄像头 ──
GET  /api/cameras         → 摄像头列表 [{index, name, default_res, ...}]
POST /api/cameras/active  → 切换摄像头 {index: 2}

── 快照/录屏 ──
POST /api/snapshot        → 保存当前帧，返回 {filename, path}
POST /api/recording       → {action: "start"|"stop"}
GET  /api/gallery         → 已保存快照列表
GET  /preview/<filename>  → 预览指定照片

── 指标 ──
GET  /api/metrics         → 瞬时指标 {fps, detect_count, latency_ms, ...}
```

### streamer.py — StreamEngine

- N 通道 MJPEG 推流，每个通道有 `label`、默认 JPEG quality、max fps
- **按需编码**：无客户端连接的通道跳过 `cv2.imencode`
- **多客户端共享**：同一 channel 多个浏览器共享同一份 JPEG 缓存
- **帧率节流**：每通道独立 maxfps 限制
- 线程安全帧缓存，采集线程写入，推流线程读取

### camera.py — CameraManager

- `detect()` — 探测所有可用摄像头，返回 `[{index, name, default_res, supported_resolutions, ...}]`
- `open(index, width, height, fps, fourcc="MJPG")` — 打开指定摄像头，返回配置好的 `cv2.VideoCapture` 实例
- 整合自旧 `detect_cameras()` 和 `camera_init()`，整合后原位置删除
- 支持 Linux 下 sysfs/V4L2 硬件树分析，过滤 IR/metadata 虚拟节点

### params.py — ParamRegistry

- 注册参数：`{name, type, default, range/min/max, step, group, description}`
- 类型校验：int / float / bool / choice
- 变更回调机制：参数被修改时触发用户注册的回调
- JSON 序列化给前端；前端提交 POST 写入
- WebSocket 广播参数变更（共享会话模型）

### recorder.py — Recorder

- 单帧快照：存为 JPG 到指定目录
- 帧序列录制：`start()` 开始积攒帧，`stop()` 合成 MP4
- 照片画廊：列出已保存文件
- 录制用 OpenCV `VideoWriter`，避免引入 ffmpeg 依赖

### log_buffer.py — LogBuffer

- 定长环形 buffer，上限 500 条
- 结构化日志：`{ts, level, tag, msg}`
- WebSocket 实时广播到前端日志面板
- 支持按 level/tag 过滤

## 前端组件

### 整体布局

```
桌面 (>768px):
┌─────────────────────────────────────────────────────┐
│  ● ● ●        OpenCV 调试面板         192.168.x.x   │ ← 毛玻璃标题栏
├──────────┬──────────────────────────────────────────┤
│ 侧边栏   │              视图网格区域                 │
│ ▸ 参数   │  ┌─────────┐  ┌─────────┐               │
│ ▸ 布局   │  │ 视图 1  │  │ 视图 2  │               │
│ ▸ 工具   │  └─────────┘  └─────────┘               │
├──────────┴──────────────────────────────────────────┤
│  ◆ FPS:58  │  检测:12  │  耗时:3.2ms  │  ● REC     │ ← 状态栏
│  ┌───────────────────────────────────────────────┐  │
│  │              实时图表 (可折叠)                 │  │
│  └───────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │              日志面板 (可折叠)                 │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘

手机 (≤768px):
┌──────────────────────┐
│ ● ● ●  OpenCV 调试   │ ← 标题栏
├──────────────────────┤
│                      │
│  ┌────────────────┐  │
│  │    视图 1      │  │ ← 单图全宽，左右滑动切通道
│  └────────────────┘  │
│  ← ● →  (页码)       │
├──────────────────────┤
│ FPS:58 │ 检:12       │ ← 缩略状态栏
├──────────────────────┤
│ [视图] [参数] [日志] │ ← 底部 Tab Bar
└──────────────────────┘
```

### 苹果风格规范

- 主背景 `#f5f5f7`，卡片 `#ffffff` + `border-radius: 18px` + 微阴影
- 字体 SF Pro 回退 `-apple-system, sans-serif`，`-webkit-font-smoothing: antialiased`
- 标题栏：`backdrop-filter: saturate(180%) blur(20px)`
- 控件：圆角滑块，细边框 toggle
- 动画：仅用 `opacity` + `transform` 过渡（GPU composite 层，不触发 layout）

### 各组件

| 组件 | 文件 | 功能 |
|------|------|------|
| 视图网格 | `layout.js` | 切换 1×1 / 1×2 / 2×2，每格选绑 stream 通道；手机端单图 + swipe |
| 参数面板 | `params.js` | 按组渲染滑块/开关/下拉，防抖 50ms 发 POST |
| ROI 工具 | `roi.js` | canvas overlay 拖拽选区，显示像素统计 |
| 实时图表 | `chart.js` | Canvas 2D 折线图，最多 120 点，增量绘制 |
| 日志面板 | `log.js` | 虚拟滚动，只渲染视口内 ~20 行 |
| 快照/录屏 | 按钮 + fetch | POST `/api/snapshot`，开始/停止录制 |

## WebSocket 协议

双向 JSON 帧：

```json
// 服务端 → 客户端
{"type": "log",     "ts": 1700000000.123, "level": "info", "tag": "DETECT", "msg": "#42 area=3421 ✓"}
{"type": "params",  "params": {"threshold": 127, "kernel": 5}}
{"type": "metrics", "fps": 58.2, "detect": 12, "latency": 3.2}
{"type": "pong"}

// 客户端 → 服务端
{"type": "set_param", "name": "threshold", "value": 130}
{"type": "ping"}
```

Metrics 每秒推送一次；params 仅在变更时广播；log 实时推送。

## 性能策略（泰山派 ARM）

### 后端

| 优化点 | 措施 |
|--------|------|
| 按需编码 | 无客户端的通道跳过 `cv2.imencode` |
| 多客户端共享 | 同通道多浏览器共享 JPEG 缓存 |
| 帧率节流 | 每通道独立 maxfps |
| JPEG 质量可配 | 手机端可请求更低质量 |
| 仅用 `.jpg` 编码 | 不用 `.png`，编码快 |
| 环形帧缓存 | 只保留最新 1 帧 |

### 前端

| 优化点 | 措施 |
|--------|------|
| 零依赖 | 不引入任何 JS/CSS 框架 |
| 流按需加载 | ≤2 路同时可见，超出暂停 img src |
| 参数防抖 | 滑块 50ms 防抖后才发请求 |
| 虚拟滚动 | 日志只渲染视口内 DOM |
| 增量绘制 | 图表 shift 画布，不每帧重绘全图 |
| CSS 动画 | 仅用 composite 层属性（transform/opacity） |

## 旧代码清理

| 现有位置 | 动作 |
|----------|------|
| `tools/tools.py` — `WebStreamer` `MJPEGHandler` | 删除 |
| `tools/tools.py` — `detect_cameras` `camera_init` | 移至 `tools/web/camera.py`，原位置删除 |
| `tools/tools.py` — 纯视觉函数（`FpsShow` `cvt_mvlab2cv` `order_points` `perspective_correct_and_validate`） | 保留原地，与 web 无关 |
| `tools/shoot.py` — 全部 | 删除 |
| `main.py` | 改为使用新 API |

## 使用示例

```python
from tools.web import DebugServer, CameraManager, ParamRegistry

# 1. 打开摄像头
cam = CameraManager()
cap = cam.open(index=9, width=1280, height=720, fps=60)  # 返回 cv2.VideoCapture

# 2. 注册可调参数
params = ParamRegistry()
params.add("threshold", type=int, default=127, range=(0, 255), group="二值化")
params.add("kernel",    type=int, default=5,   range=(1, 21), step=2, group="形态学")

# 3. 启动 web 服务（非阻塞，HTTP 在后台线程）
server = DebugServer(params=params, port=8080)
server.start()

# 4. 主循环
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    processed = user_pipeline(frame, params.snapshot())

    server.update_frame(0, frame)       # 通道 0: 原始帧
    server.update_frame(1, processed)   # 通道 1: 处理结果
    server.log("DETECT", f"#42 area={area}")
    server.metrics.update(fps=58.2, detect=12)

# 5. 退出（可选，程序结束时自动清理）
server.stop()
cap.release()
```
