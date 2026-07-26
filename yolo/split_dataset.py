#!/usr/bin/env python3
"""YOLO 数据集拆分与校验脚本。

读取原始图片+标注目录，校验缺漏、统计类别框数，按指定比例随机拆分为
train/val/test 三个子集，并生成 Ultralytics 格式的 data.yaml。

用法:
    python scripts/split_dataset.py --raw yolo/raw --out yolo --ratio 7:2:1
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO 数据集拆分与校验")
    p.add_argument(
        "--raw", required=True,
        help="原始数据根目录（含 images/ 和 labels/ 子目录）",
    )
    p.add_argument(
        "--out", required=True,
        help="输出根目录（生成 train/val/test 子目录及 data.yaml）",
    )
    p.add_argument(
        "--ratio", default="7:2:1",
        help="train:val:test 比例（默认 7:2:1）",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42）",
    )
    return p.parse_args()


def _print_preview(label: str, items: list[str], limit: int = 8) -> None:
    """打印标签 + 数量 + 预览（前 N 项）。"""
    print(f"  {label}: {len(items)}")
    if items:
        preview = ", ".join(items[:limit])
        suffix = " ..." if len(items) > limit else ""
        print(f"    → {preview}{suffix}")


def _find_class_names(raw: Path) -> dict[int, str]:
    """从多个可能位置查找类名文件，返回 {class_id: name} 映射。"""
    candidates = [
        raw / "classes.txt",
        raw / "class.txt",
        raw / "labels" / "classes.txt",
        raw / "labels" / "class.txt",
    ]
    for fpath in candidates:
        if fpath.exists():
            lines = [line.strip() for line in fpath.read_text().strip().splitlines() if line.strip()]
            return {i: name for i, name in enumerate(lines)}
    return {}


def convert_labelme_to_yolo(raw: Path) -> None:
    """将 labelme JSON 标注转换为 YOLO TXT。

    仅处理 shape_type == "rectangle" 的标注。
    非 rectangle 形状跳过并打印 warning。
    原始 JSON 文件保留不动。
    """
    lbl_dir = raw / "labels"
    json_paths = sorted(lbl_dir.glob("*.json"))
    if not json_paths:
        print("[convert] 未发现 labelme JSON 标注文件")
        return

    class_names = _find_class_names(raw)
    if not class_names:
        raise ValueError("未找到 classes.txt，无法确定类别映射")
    # 建立 label 字符串 → class_id 的反向映射
    label_to_id: dict[str, int] = {name: i for i, name in class_names.items()}

    converted = 0
    skipped = 0
    warnings: list[str] = []

    for json_path in json_paths:
        txt_path = json_path.with_suffix(".txt")
        if txt_path.exists():
            skipped += 1
            continue

        data = json.loads(json_path.read_text())
        if "imageWidth" not in data or "imageHeight" not in data:
            raise ValueError(f"JSON 缺少 imageWidth/imageHeight 字段: {json_path}")
        width = data["imageWidth"]
        height = data["imageHeight"]
        if width <= 0 or height <= 0:
            raise ValueError(f"JSON 图片尺寸无效 ({width}x{height}): {json_path}")

        lines: list[str] = []
        for shape in data.get("shapes", []):
            shape_type = shape.get("shape_type", "")
            if shape_type != "rectangle":
                warnings.append(
                    f"[warn] 跳过非矩形标注: {json_path.name} "
                    f"shape_type={shape_type}"
                )
                continue

            label = (shape.get("label") or "").strip()
            if not label:
                raise ValueError(f"标注 label 为空: {json_path}")
            if label not in label_to_id:
                raise ValueError(
                    f"标注 label '{label}' 不在 classes.txt 中: {json_path}"
                )

            points = shape.get("points", [])
            if len(points) < 2:
                raise ValueError(f"矩形标注缺少有效顶点: {json_path}")
            # labelme 矩形顶点的点击顺序可能为任意对角方向，取 min/max
            x1 = min(p[0] for p in points[:2])
            y1 = min(p[1] for p in points[:2])
            x2 = max(p[0] for p in points[:2])
            y2 = max(p[1] for p in points[:2])

            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(
                    f"标注框越界或尺寸无效: {json_path} "
                    f"({x1}, {y1}, {x2}, {y2}) / ({width}, {height})"
                )

            x_center = (x1 + x2) / (2 * width)
            y_center = (y1 + y2) / (2 * height)
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            lines.append(
                f"{label_to_id[label]} {x_center:.6f} {y_center:.6f} "
                f"{box_width:.6f} {box_height:.6f}"
            )

        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        converted += 1

    for w in warnings:
        print(w)
    print(
        f"[convert] labelme JSON → YOLO: {converted} 个文件已转换，"
        f"{skipped} 个已有 TXT 已跳过"
    )


def validate(raw: Path) -> tuple[list[str], dict[str, int], dict[str, str]]:
    """校验图片与标注配对。

    Returns:
        (有效配对 stem 列表, 各类别框数, stem→扩展名映射)。
    """
    img_dir = raw / "images"
    lbl_dir = raw / "labels"

    if not img_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {img_dir}")
    if not lbl_dir.is_dir():
        raise FileNotFoundError(f"标注目录不存在: {lbl_dir}")

    # 收集所有图片 stem（处理同一 stem 多扩展名的情况）
    stem_to_ext: dict[str, str] = {}
    for f in img_dir.iterdir():
        if f.suffix.lower() in IMG_EXTS:
            if f.stem in stem_to_ext:
                print(f"[warn] 同名图片存在多扩展名: {f.stem}，使用 {f.suffix}")
            stem_to_ext[f.stem] = f.suffix

    img_stems = set(stem_to_ext.keys())
    lbl_stems = {f.stem for f in lbl_dir.glob("*.txt")}

    missing_labels = sorted(img_stems - lbl_stems)
    missing_images = sorted(lbl_stems - img_stems)
    empty_labels: list[str] = []
    class_counts: dict[str, int] = defaultdict(int)

    pairs: list[str] = []
    for stem in sorted(img_stems & lbl_stems):
        lbl_path = lbl_dir / f"{stem}.txt"
        content = lbl_path.read_text().strip()
        if not content:
            empty_labels.append(stem)
        else:
            pairs.append(stem)
            for line in content.splitlines():
                cls_id = line.split()[0]
                class_counts[cls_id] += 1

    # ── 打印校验报告 ──
    print("=" * 55)
    print("  数据集校验报告")
    print("=" * 55)
    print(f"  图片文件数:         {len(img_stems)}")
    print(f"  标注文件数:         {len(lbl_stems)}")
    print(f"  有效配对:           {len(pairs)}")
    _print_preview("缺标注的图片", missing_labels)
    _print_preview("缺图片的标注", missing_images)
    _print_preview("空标注（无框）", empty_labels)

    if class_counts:
        print(f"\n  类别框数统计:")
        for cls_id, count in sorted(class_counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
            print(f"    class {cls_id}: {count} boxes")
    else:
        print(f"\n  [warn] 没有检测到任何标注框")

    print("=" * 55)
    return pairs, dict(class_counts), stem_to_ext


def split_and_copy(
    pairs: list[str],
    stem_to_ext: dict[str, str],
    raw: Path,
    out: Path,
    ratio: str,
    seed: int,
    class_names: dict[int, str],
) -> None:
    """按比例拆分并复制图片+标注，生成 data.yaml。"""
    r = [int(x) for x in ratio.split(":")]
    total = sum(r)
    train_frac, val_frac, test_frac = r[0] / total, r[1] / total, r[2] / total

    rng = random.Random(seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    img_dir = raw / "images"
    lbl_dir = raw / "labels"

    for split_name, stems in splits.items():
        out_img = out / split_name / "images"
        out_lbl = out / split_name / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        copied = 0
        for stem in stems:
            ext = stem_to_ext[stem]
            src_img = img_dir / f"{stem}{ext}"
            if src_img.exists():
                shutil.copy2(src_img, out_img / src_img.name)
            else:
                print(f"[warn] 图片文件不存在: {src_img}")
                continue
            src_lbl = lbl_dir / f"{stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, out_lbl / f"{stem}.txt")
            copied += 1

        print(f"  {split_name}: {copied} images / {n} total")

    # 生成 data.yaml
    nc = len(class_names) if class_names else _infer_nc(raw)
    names = class_names if class_names else {i: f"class_{i}" for i in range(nc)}

    data_yaml = out / "data.yaml"
    yaml_content = f"""# Auto-generated by split_dataset.py — do not edit manually.
path: {out.resolve()}
train: train/images
val: val/images
test: test/images

nc: {nc}
names: {names}
"""
    data_yaml.write_text(yaml_content)
    print(f"\n  data.yaml → {data_yaml}")
    print(f"  nc={nc}, names={names}")


def _infer_nc(raw: Path) -> int:
    """从标注文件中推断类别数（找最大 class_id + 1）。"""
    lbl_dir = raw / "labels"
    max_cls = -1
    for lbl in lbl_dir.glob("*.txt"):
        for line in lbl.read_text().strip().splitlines():
            if not line.strip():
                continue
            try:
                cls_id = int(line.split()[0])
                max_cls = max(max_cls, cls_id)
            except ValueError:
                pass
    return max_cls + 1 if max_cls >= 0 else 0


def main() -> None:
    args = parse_args()
    raw = Path(args.raw).resolve()
    out = Path(args.out).resolve()

    # 1. 自动转换 labelme JSON 标注为 YOLO TXT
    convert_labelme_to_yolo(raw)

    # 2. 查找类名
    class_names = _find_class_names(raw)

    # 3. 校验 + 统计
    pairs, class_counts, stem_to_ext = validate(raw)

    if not pairs:
        print("\n[error] 没有找到任何有效配对，无法拆分。")
        raise SystemExit(1)

    # 4. 拆分 + 复制 + 生成 data.yaml
    print(f"\n  拆分比例: {args.ratio}  (seed={args.seed})")
    split_and_copy(pairs, stem_to_ext, raw, out, args.ratio, args.seed, class_names)

    # 5. 最终汇总
    print(f"\n  拆分完成！输出目录: {out}")
    print(f"  下一步: 编辑 {out}/data.yaml 确认类别名，然后 python yolo/train.py")


if __name__ == "__main__":
    main()
