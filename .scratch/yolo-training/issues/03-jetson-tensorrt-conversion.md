# 03 — Jetson 端 TensorRT 引擎转换

**What to build:** Jetson Orin NX 上的 TensorRT 转换脚本，使用 JetPack 自带的 `trtexec` 将训练端产出的 ONNX 模型转换为 FP16 TensorRT `.engine` 文件，放入 `models/` 目录供推理接口加载。

**Blocked by:** 02 — 训练脚本 + ONNX 导出（需要 best.onnx）。

**Status:** completed

- [ ] `scripts/export_tensorrt.sh` 存在，接收 `<input.onnx> <output.engine>` 参数
- [ ] 使用 `trtexec --fp16 --onnx=... --saveEngine=...` 转换
- [ ] 设置 `minShapes`/`optShapes`/`maxShapes` 为 `images:1x3x640x640`
- [ ] 在 Jetson Orin NX 上执行成功，生成 `.engine` 文件
- [ ] 用 `trtexec --loadEngine=...` 验证 engine 可加载且推理输出形状正确
