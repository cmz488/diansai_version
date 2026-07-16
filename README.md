# Orin NX OpenCV 视觉管线

默认数据流：

```text
USB MJPEG camera
  -> nvv4l2decoder (NVDEC)
  -> nvvidconv compute-hw=2 (VIC: flip/format conversion)
  -> OpenCV BGR ndarray
  -> project detection
```

## 运行

```bash
cd /home/amov/Project

# 默认优先 NVDEC+VIC，失败时回退 V4L2。
python3 main.py

# 仅测试视频流帧率。
python3 test_fps.py --backend auto --frames 300

# 测试所有可用的采集/计算组合并选择最高 FPS。
python3 benchmark_hardware.py --frames 300 --warmup 30
```

常用参数：

```bash
python3 main.py \
  --device /dev/video0 \
  --width 1280 --height 720 --fps 60 \
  --capture-backend nvdec_vic \
  --compute-backend cpu
```

## OpenCV 交互接口

```python
from tools.hardware_pipeline import OpenCVHardwarePipeline, PipelineConfig

pipeline = OpenCVHardwarePipeline(
    PipelineConfig(device="/dev/video0", width=1280, height=720, fps=60),
    capture_backend="auto",
    compute_backend="auto",  # 当前端到端实测选择 CPU 检测链
)

with pipeline:
    ok, frame = pipeline.read()  # frame 是普通 OpenCV BGR ndarray
```

`OpenCVHardwarePipeline.preprocess()` 与原 `preprocess()` 返回相同的
`(rect_edges, laser_binary, gray)`，业务代码不需要接触 GStreamer、VPI
或 CUDA 对象。

## 后端说明

- `nvdec_vic`：USB MJPEG 硬件解码 + VIC 翻转/转换，是本机端到端最优路径。
- `v4l2`：CPU/驱动回退路径，接口相同。
- `cpu`：OpenCV CPU 检测链；当前 60 FPS 相机的端到端结果略优，故为默认。
- `vpi_cuda`：把 Gaussian/Canny/闭运算放到 VPI CUDA，并复用 VPI 缓冲区。
  当前长测算法吞吐约 116 FPS，CPU 约 99 FPS；60 FPS 相机下端到端基本持平。
- `umat`：只在 `cv2.ocl.useOpenCL() == True` 时允许。当前 Jetson
  OpenCV 只有 OpenCL loader、没有 OpenCL 设备实现，UMat 会退化为 CPU，
  因此默认不会把它标记成硬件加速。
- PyCUDA 2022.2.2 已与系统 NumPy/OpenCV 兼容，基准脚本会执行真实 CUDA
  kernel 自检；它不被强行塞入 LAB/轮廓管线，因为主机和 GPU 往返会降低
  当前端到端 FPS。

OpenCV appsink 最终仍有一次 NVMM 到 CPU BGR 的复制。若未来需要多路
1080p/4K 完整零拷贝，应迁移到 C++ NvBufSurface/DeepStream，而不是在
Python 中每帧下载到 ndarray。

## 项目级 DNS 修复

系统 `/etc/resolv.conf` 指向了不存在的
`/run/systemd/resolve/stub-resolve.conf`。没有 root 权限时，可在私有
mount namespace 中运行联网命令：

```bash
./run_with_dns.sh curl -I https://docs.nvidia.com
./run_with_dns.sh pip3 install --user <package>
```

该脚本不修改系统全局文件。永久修复仍需管理员把 `/etc/resolv.conf`
链接到 `/run/systemd/resolve/stub-resolv.conf`。
