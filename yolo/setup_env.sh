#!/usr/bin/env bash
# 从 environment.yml 干净重建 RTX 5060 YOLO 训练环境。
set -euo pipefail

ENV_NAME="yolo"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment.yml"

if ! command -v conda >/dev/null 2>&1; then
    echo "错误：找不到 conda，请先安装 Miniconda 或 Miniforge。" >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ "${CONDA_DEFAULT_ENV:-}" == "${ENV_NAME}" ]]; then
    echo "=== 退出当前 ${ENV_NAME} 环境 ==="
    conda deactivate
fi

if conda env list | grep -Eq "^${ENV_NAME}[[:space:]]"; then
    echo "=== 删除现有 ${ENV_NAME} 环境 ==="
    conda env remove --name "${ENV_NAME}" --yes
fi

echo "=== 从 ${ENV_FILE} 创建 ${ENV_NAME} 环境 ==="
conda env create --file "${ENV_FILE}"

echo "=== 验证 PyTorch、RTX 5060 与主要依赖 ==="
conda run --no-capture-output --name "${ENV_NAME}" python - <<'PY'
import torch
import ultralytics
import onnx
import onnxruntime
import cv2
import torchvision

expected = {
    "torch": "2.13.0+cu130",
    "torchvision": "0.28.0+cu130",
    "cuda": "13.0",
}
actual = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda": torch.version.cuda,
}
for key, expected_value in expected.items():
    if actual[key] != expected_value:
        raise RuntimeError(f"{key} 版本错误：期望 {expected_value}，实际 {actual[key]}")

arches = torch.cuda.get_arch_list()
if "sm_120" not in arches:
    raise RuntimeError(f"PyTorch wheel 未包含 RTX 5060 所需的 sm_120：{arches}")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch 无法访问 CUDA；请检查 nvidia-smi 和 NVIDIA 设备节点")

device_name = torch.cuda.get_device_name(0)
if "RTX 5060" not in device_name:
    raise RuntimeError(f"GPU 不是预期的 RTX 5060：{device_name}")

# 执行一次真实的 CUDA 运算，避免只验证到设备枚举。
x = torch.ones((256, 256), device="cuda")
y = x @ x
torch.cuda.synchronize()
if y[0, 0].item() != 256.0:
    raise RuntimeError("CUDA 矩阵乘法结果异常")

print(f"PyTorch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"Compiled CUDA architectures: {arches}")
print(f"GPU: {device_name}")
print(f"Ultralytics: {ultralytics.__version__}")
print(f"ONNX: {onnx.__version__}")
print(f"ONNX Runtime: {onnxruntime.__version__}")
print(f"OpenCV: {cv2.__version__}")
print("CUDA matrix multiplication: OK")
PY

echo ""
echo "=== 环境安装完成 ==="
echo "以后使用时执行：conda activate ${ENV_NAME}"
