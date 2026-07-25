#!/usr/bin/env python3
"""在独立 test 数据集上评估训练后的 YOLO 模型。

默认自动选择 ``runs/detect/**/weights/best.pt`` 中最新的权重，计算
Precision、Recall、mAP50 和 mAP50-95，并保存混淆矩阵、PR/F1 曲线及
预测可视化图。

用法::

    conda activate yolo
    python tests/test_yolo_model.py
    python tests/test_yolo_model.py --weights runs/detect/train-2/weights/best.pt
    python tests/test_yolo_model.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "yolo" / "data.yaml"
DEFAULT_RUNS = PROJECT_ROOT / "runs" / "detect"
TRAIN_CONFIG = PROJECT_ROOT / "yolo" / "train_config.yaml"


def project_path(value: str | Path) -> Path:
    """将相对路径按项目根目录解析。"""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def find_latest_best(runs_dir: Path) -> Path:
    """查找最近修改的训练最佳权重。"""
    candidates = list(runs_dir.glob("**/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"在 {runs_dir} 下找不到 best.pt，请先训练或使用 --weights 指定权重"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_train_defaults() -> dict:
    """复用训练配置中的推理尺寸、批量和设备设置。"""
    if not TRAIN_CONFIG.exists():
        return {}
    content = yaml.safe_load(TRAIN_CONFIG.read_text())
    return content if isinstance(content, dict) else {}


def validate_data_config(data_path: Path) -> None:
    """在启动耗时评估前确认 data.yaml 声明了 test 集。"""
    if not data_path.is_file():
        raise FileNotFoundError(f"找不到数据集配置: {data_path}")
    data = yaml.safe_load(data_path.read_text())
    if not isinstance(data, dict) or not data.get("test"):
        raise ValueError(f"数据集配置未声明 test 路径: {data_path}")
    if not data.get("names"):
        raise ValueError(f"数据集配置未声明 names 类别: {data_path}")


def parse_args() -> argparse.Namespace:
    defaults = load_train_defaults()
    parser = argparse.ArgumentParser(description="评估 YOLO 模型在 test 数据集上的效果")
    parser.add_argument(
        "--weights",
        help="训练权重路径；默认自动选择 runs/detect 下最新的 best.pt",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help=f"数据集配置（默认: {DEFAULT_DATA}）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(defaults.get("imgsz", 640)),
        help="评估输入尺寸",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=int(defaults.get("batch", 16)),
        help="评估批量大小",
    )
    parser.add_argument(
        "--device",
        default=str(defaults.get("device", "0")),
        help="设备，例如 0、0,1 或 cpu",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(defaults.get("workers", 4)),
        help="DataLoader 进程数",
    )
    parser.add_argument(
        "--project",
        default=str(DEFAULT_RUNS),
        help="评估结果根目录",
    )
    parser.add_argument("--name", default="test", help="评估运行名称")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="使用 FP16 评估",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="生成混淆矩阵和 PR/F1 曲线",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = project_path(args.data)
    project_dir = project_path(args.project)
    weights_path = (
        project_path(args.weights)
        if args.weights
        else find_latest_best(DEFAULT_RUNS)
    )

    if not weights_path.is_file():
        raise FileNotFoundError(f"找不到模型权重: {weights_path}")
    validate_data_config(data_path)

    from ultralytics import YOLO
    from ultralytics.utils.files import increment_path
    import torch

    device = args.device
    numeric_device = device.replace(",", "").isdigit()
    if numeric_device and not torch.cuda.is_available():
        print(f"[warn] CUDA 设备 {device} 不可用，自动回退到 CPU")
        device = "cpu"

    print(f"[test] weights: {weights_path}")
    print(f"[test] data:    {data_path}")
    print(
        f"[test] split=test, imgsz={args.imgsz}, batch={args.batch}, "
        f"device={device}"
    )

    save_dir = increment_path(project_dir / args.name, exist_ok=False).resolve()
    model = YOLO(str(weights_path))
    val_args = dict(
        data=str(data_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
        plots=args.plots,
    )
    if args.half:
        val_args["half"] = True
    metrics = model.val(**val_args)

    results = {key: float(value) for key, value in metrics.results_dict.items()}
    summary = {
        "weights": str(weights_path),
        "data": str(data_path),
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "metrics": results,
    }
    summary_path = save_dir / "test_metrics.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )

    labels = {
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50-95",
    }
    print("\n" + "=" * 52)
    print("  YOLO test 集评估结果")
    print("=" * 52)
    for key, label in labels.items():
        if key in results:
            print(f"  {label:<10}: {results[key]:.4f}")
    print("=" * 52)
    print(f"[ok] 图表与详细结果: {save_dir}")
    print(f"[ok] JSON 指标摘要: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
