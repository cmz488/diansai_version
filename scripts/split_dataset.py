#!/usr/bin/env python3
"""YOLO 数据集拆分与校验脚本。

读取原始图片+标注目录，校验缺漏、统计类别框数，按指定比例随机拆分为
train/val/test 三个子集，并生成 Ultralytics 格式的 data.yaml。

用法:
    python scripts/split_dataset.py --raw yolo/raw --out yolo --ratio 7:2:1
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

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


def validate(raw: Path) -> tuple[list[str], dict[str, int]]:
    """校验图片与标注配对，返回 (有效配对 stem 列表, 各类别框数)。"""
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
    print(f"  缺标注的图片:       {len(missing_labels)}")
    if missing_labels:
        preview = ", ".join(missing_labels[:8])
        suffix = " ..." if len(missing_labels) > 8 else ""
        print(f"    → {preview}{suffix}")
    print(f"  缺图片的标注:       {len(missing_images)}")
    if missing_images:
        preview = ", ".join(missing_images[:8])
        suffix = " ..." if len(missing_images) > 8 else ""
        print(f"    → {preview}{suffix}")
    print(f"  空标注（无框）:     {len(empty_labels)}")
    if empty_labels:
        preview = ", ".join(empty_labels[:8])
        suffix = " ..." if len(empty_labels) > 8 else ""
        print(f"    → {preview}{suffix}")

    if class_counts:
        print(f"\n  类别框数统计:")
        for cls_id, count in sorted(class_counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
            print(f"    class {cls_id}: {count} boxes")
    else:
        print(f"\n  [warn] 没有检测到任何标注框")

    print("=" * 55)
    return pairs, dict(class_counts)


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

    # 1. 查找类名
    class_names = _find_class_names(raw)

    # 2. 校验 + 统计
    pairs, class_counts = validate(raw)

    if not pairs:
        print("\n[error] 没有找到任何有效配对，无法拆分。")
        raise SystemExit(1)

    # 3. 收集 stem→ext 映射（用于复制时找到正确的扩展名）
    img_dir = raw / "images"
    stem_to_ext: dict[str, str] = {}
    for f in img_dir.iterdir():
        if f.suffix.lower() in IMG_EXTS:
            stem_to_ext[f.stem] = f.suffix

    # 4. 拆分 + 复制 + 生成 data.yaml
    print(f"\n  拆分比例: {args.ratio}  (seed={args.seed})")
    split_and_copy(pairs, stem_to_ext, raw, out, args.ratio, args.seed, class_names)

    # 5. 最终汇总
    print(f"\n  拆分完成！输出目录: {out}")
    print(f"  下一步: 编辑 {out}/data.yaml 确认类别名，然后 python yolo/train.py")


if __name__ == "__main__":
    main()
