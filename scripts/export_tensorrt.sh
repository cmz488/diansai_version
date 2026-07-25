#!/bin/bash
# Jetson Orin NX / AllSpark2: ONNX → TensorRT FP16 Engine
#
# 依赖: JetPack 自带的 trtexec（无需额外安装）
#
# 用法:
#   bash scripts/export_tensorrt.sh <input.onnx> <output.engine>
#
# 示例:
#   bash scripts/export_tensorrt.sh models/best.onnx models/yolov11n_best.engine

set -euo pipefail

INPUT="${1:?Usage: $0 <input.onnx> <output.engine>}"
OUTPUT="${2:?}"

if [[ ! -f "$INPUT" ]]; then
    echo "[error] ONNX 文件不存在: $INPUT"
    exit 1
fi

echo "============================================"
echo "  ONNX → TensorRT FP16 Engine"
echo "============================================"
echo "  Input:  $INPUT"
echo "  Output: $OUTPUT"
echo ""

# 动态 batch 暂不使用，固定 batch=1 以获得最佳推理性能
trtexec \
    --onnx="${INPUT}" \
    --saveEngine="${OUTPUT}" \
    --fp16 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x640x640

echo ""
echo "============================================"
echo "  Engine saved: $OUTPUT"
echo "============================================"

# 验证 engine 可加载
echo "[verify] 验证 engine 推理..."
trtexec --loadEngine="${OUTPUT}" --iterations=1 2>&1 | tail -5

echo ""
echo "[ok] TensorRT engine 转换完成"
echo "  推理接口: YoloDetector(engine_path='${OUTPUT}')"
