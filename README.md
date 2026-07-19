# Orin NX CSI/USB OpenCV 视觉识别

项目默认使用 Jetson CSI 摄像头链路：

```text
CSI sensor
  -> nvarguscamerasrc (Argus + ISP)
  -> nvvidconv compute-hw=2 (VIC: flip/format conversion)
  -> BGRx -> BGR appsink
  -> OpenCV BGR ndarray
  -> project detection
```

原有 USB MJPEG -> NVDEC -> VIC 链路仍可用，通过 `--source usb` 选择。
硬件管线不可用时程序直接报错，不会静默切换到性能不同的软件路径。

## 项目结构

```text
main.py                    应用入口：采集、矩形检测、Web 调试输出
tools/hardware_pipeline.py CSI Argus/VIC 与 USB NVDEC/VIC 相机封装
tools/tools.py             OpenCV 预处理和检测算法
tools/web/                 参数调节与画面调试服务
test_fps.py                固定采集链路的 FPS 测试
benchmark_hardware.py      固定采集链路的延迟统计
```

## 运行

```bash
cd /home/amov/Project
python3 main.py --source csi --sensor-id 0 --width 1280 --height 720 --fps 60
```

可选参数：

```text
--flip-method 0..7   VIC 翻转方式，默认 6
--sensor-mode N      Argus 传感器模式，默认 -1 自动选择
--port 8080          Web 调试服务端口
--no-web             不启动 Web 调试服务
--max-frames N       处理 N 帧后退出，0 表示持续运行
```

## 最小采集接口

```python
from tools.hardware_pipeline import JetsonCamera, PipelineConfig

camera = JetsonCamera(
    PipelineConfig(source="csi", sensor_id=0, width=1280, height=720, fps=60)
)

with camera:
    ok, frame = camera.read()
    # frame: H x W x 3 的 OpenCV BGR ndarray
```

`JetsonCamera` 只负责把硬件处理后的画面交给 OpenCV；颜色阈值、边缘提取
和目标识别仍由项目检测代码负责。

使用 USB MJPEG 摄像头时：

```bash
python3 main.py --source usb --device /dev/video0 --width 1280 --height 720 --fps 60
```

## CSI 驱动检查

Argus 能打开摄像头之前，系统必须先出现 `/dev/video*`。常用检查命令：

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=30 \
  ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1' \
  ! nvvidconv ! fakesink
```

若没有 `/dev/video*`，应先修复传感器型号、CSI 接口对应的设备树 overlay、
排线方向或接触问题；此时修改 OpenCV 参数无效。

## 性能检查

```bash
python3 test_fps.py --source csi --frames 300 --warmup 30
python3 benchmark_hardware.py --source csi --frames 300 --warmup 30
```
