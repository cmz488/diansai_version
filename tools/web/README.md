# OpenCV Web Debugger 使用说明

`tools/web` 是一个 OpenCV 调试与展示的 Web 面板，支持桌面和手机浏览器。

## 快速开始

```python
from tools.web import DebugServer, ParamRegistry, CameraManager

# 1. 注册可调参数
params = ParamRegistry()
params.add("threshold", type=int, default=127, range=(0, 255), group="二值化")
params.add("kernel",    type=int, default=5,   range=(1, 21), step=2, group="形态学")
params.add("debug",     type=bool, default=False, group="开关")
params.add("mode",      type="choice", default="fast", choices=["fast", "slow"], group="模式")

# 2. 启动 Web 服务（非阻塞）
server = DebugServer(params=params, port=8080)
server.start()
print("浏览器打开 http://<设备IP>:8080")

# 3. 主循环
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    # 读取当前参数
    thresh = params.get("threshold")

    # 处理
    processed = your_pipeline(frame, thresh)

    # 推流到网页（最多 8 个通道）
    server.update_frame(0, frame)       # 通道 0: 原始帧
    server.update_frame(1, processed)   # 通道 1: 处理结果

    # 写日志（网页实时显示）
    server.log("DETECT", "发现目标, area=3421")
    server.log("REJECT", "面积太小", "warn")

    # 更新指标（FPS 图表数据源）
    server.metrics.update(fps=58.2, detect=12, latency=3.2)
    server.broadcast_metrics()
```

## API 参考

### DebugServer

| 方法 | 说明 |
|------|------|
| `DebugServer(params, port, host, save_dir)` | 构造函数，所有参数可选 |
| `.start()` | 非阻塞启动，HTTP 在后台 daemon 线程 |
| `.stop()` | 安全关闭 |
| `.update_frame(channel_id, frame)` | 推送 OpenCV 帧到通道（0-7） |
| `.log(tag, msg, level="info")` | 写结构化日志，level: info/warn/error |
| `.metrics.update(**kv)` | 更新指标，推送到前端图表 |
| `.broadcast_metrics()` | 定时调用（建议每帧），推送指标到所有客户端 |

### ParamRegistry

| 方法 | 说明 |
|------|------|
| `.add(name, type, default, range, step, choices, group)` | 注册参数 |
| `.get(name)` | 读取当前值 |
| `.set(name, value)` | 更新值（会校验类型/范围） |
| `.snapshot()` | 返回 `{name: value}` |
| `.on_change(callback)` | 参数变更回调 |

**参数类型：**

| type | 说明 | 前端控件 | 额外字段 |
|------|------|----------|----------|
| `int` | 整数 | 滑块 | `range=(0, 255)` |
| `float` | 浮点数 | 滑块 | `range=(0.0, 1.0)`, `step=0.01` |
| `bool` | 开关 | 复选框 | - |
| `choice` | 下拉选择 | 下拉框 | `choices=["a", "b"]` |

### CameraManager

```python
cam = CameraManager()
cams = cam.detect()                          # 探测所有摄像头
cap  = cam.open(0, width=1280, height=720)   # 打开并配置
```

## 网页面板说明

打开 `http://<IP>:8080` 后：

### 桌面端布局

```
┌──────────────────────────────────────────┐
│  ● ● ●    OpenCV 调试面板    192.168.x.x │  标题栏
├────────────┬─────────────────────────────┤
│  侧边栏    │         视图网格             │
│            │  ┌────────┐ ┌────────┐     │
│ 参数滑块   │  │ 原始帧 │ │ 边缘图 │     │
│ 布局切换   │  └────────┘ └────────┘     │
│ 工具按钮   │                             │
├────────────┴─────────────────────────────┤
│  ◆ FPS:58  │  检测:12  │  耗时:3.2ms    │  状态栏
│  ┌─────────────────────────────────┐    │
│  │   📊 图表 (点击切换)             │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │   📋 日志 (点击切换)             │    │
│  └─────────────────────────────────┘    │
└──────────────────────────────────────────┘
```

- **侧边栏**：调节参数滑块实时生效，切换 1×1 / 1×2 / 2×2 布局
- **视图网格**：每个格子显示一路视频流。1×1 模式下点击格子可切换通道
- **图表面板**：实时折线图，显示 FPS、检测数等指标
- **日志面板**：后端推送的结构化日志，info/warn/error 分级颜色

### 手机端布局

```
┌─────────────────────┐
│ ● ● ●  OpenCV 调试   │  标题栏
├─────────────────────┤
│  ┌─────────────────┐│
│  │   视频流 (单路)  ││  左右滑动切通道
│  └─────────────────┘│
├─────────────────────┤
│ FPS:58  │  检测:12  │  状态栏
├─────────────────────┤
│ [视图] [参数] [日志] │  底部 Tab 栏
└─────────────────────┘
```

- **视图 Tab**：单路视频，左右滑动切换通道
- **参数 Tab**：弹出式面板，调节滑块
- **日志 Tab**：弹出式面板，查看日志（长消息自动换行）

### 工具栏按钮

| 按钮 | 功能 |
|------|------|
| **📐 ROI 选区** | 在图像上拖拽矩形区域 |
| **📸 快照** | 保存当前帧为 JPG，存到 `./photos/` |
| **⏺ 录屏** | 开始/停止 MP4 录制 |

## StreamEngine（高级用法）

如果不需要整套 DebugServer，可以单独用 StreamEngine 推多路 MJPEG：

```python
from tools.web.streamer import StreamEngine

engine = StreamEngine(max_channels=4)
engine.configure(0, label="原始帧", quality=70, maxfps=30)

# 主循环
engine.update(0, frame)

# 在自己的 HTTP 服务器里用 engine.get_jpeg(0) 获取 JPEG bytes
```

## 配置 MJPEG 流

通过 URL query string 控制单个流的编码参数：

- `http://IP:8080/stream/0?quality=50` — 更低 JPEG 质量（省带宽）
- `http://IP:8080/stream/0?maxfps=15` — 限制帧率（省 CPU）

## 注意事项

- 泰山派 ARM 设备建议 JPEG quality ≤ 70，maxfps ≤ 30
- 手机端同时只解码 1 路流（CPU 优化），桌面端最多 2 路
- 照片和录像保存到 `./photos/` 目录
