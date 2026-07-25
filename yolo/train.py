#!/usr/bin/env python3
"""YOLO 训练入口 — 读取 train_config.yaml，零参数启动训练 + ONNX 导出。

用法:
    conda activate yolo
    python yolo/train.py
"""

import sys
from pathlib import Path

import yaml


def find_config() -> Path:
    """在 yolo/ 目录下查找 train_config.yaml。"""
    candidates = [
        Path("yolo/train_config.yaml"),
        Path("train_config.yaml"),
        Path(__file__).resolve().parent / "train_config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    print("[error] 找不到 train_config.yaml", file=sys.stderr)
    raise SystemExit(1)


def load_config(path: Path) -> dict:
    """加载 YAML 配置，校验必填字段。"""
    cfg = yaml.safe_load(path.read_text())
    required = ["data", "model", "epochs", "batch", "imgsz"]
    missing = [k for k in required if k not in cfg]
    if missing:
        print(f"[error] train_config.yaml 缺少字段: {missing}", file=sys.stderr)
        raise SystemExit(1)
    return cfg


def train(cfg: dict) -> Path:
    """执行训练，返回 best.pt 路径。"""
    from ultralytics import YOLO

    print(f"[train] model: {cfg['model']}")
    print(f"[train] data:  {cfg['data']}")
    print(f"[train] epochs={cfg['epochs']}, batch={cfg['batch']}, imgsz={cfg['imgsz']}")

    model = YOLO(cfg["model"])
    model.train(
        data=cfg["data"],
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        imgsz=cfg["imgsz"],
        # 其余参数用 Ultralytics 默认值
    )

    best_path = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        print(f"[error] 训练完成但找不到 best.pt: {best_path}", file=sys.stderr)
        raise SystemExit(1)

    print(f"[train] best model: {best_path}")
    return best_path


def export_onnx(pt_path: Path) -> Path:
    """导出 best.pt 为 ONNX。"""
    from ultralytics import YOLO

    onnx_path = pt_path.with_suffix(".onnx")
    print(f"[export] {pt_path} → {onnx_path}")

    model = YOLO(str(pt_path))
    model.export(format="onnx", imgsz=640, opset=12, simplify=True)

    if not onnx_path.exists():
        print(f"[error] ONNX 导出失败，文件未生成: {onnx_path}", file=sys.stderr)
        raise SystemExit(1)

    # 验证 ONNX 模型
    try:
        import onnx
        onnx.checker.check_model(str(onnx_path))
        print("[export] ONNX model check passed")
    except ImportError:
        print("[warn] onnx 包未安装，跳过 check_model")
    except Exception as e:
        print(f"[warn] ONNX check_model 失败: {e}")

    print(f"[export] done: {onnx_path}")
    return onnx_path


def main() -> None:
    config_path = find_config()
    print(f"[init] config: {config_path}")

    cfg = load_config(config_path)

    best_pt = train(cfg)
    export_onnx(best_pt)

    print("\n[ok] 训练 + ONNX 导出完成")
    print(f"  PyTorch: {best_pt}")
    print(f"  ONNX:    {best_pt.with_suffix('.onnx')}")
    print(f"  下一步: scp ONNX 到 Jetson，运行 scripts/export_tensorrt.sh")


if __name__ == "__main__":
    main()
