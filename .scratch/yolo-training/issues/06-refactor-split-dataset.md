# 06 — split_dataset.py DRY 重构

**What to build:** 两项纯重构，不改变对外行为：
1. 提取 `_print_preview(label, items)` 辅助函数，消除 `validate()` 中 `missing_labels`/`missing_images`/`empty_labels` 三处重复的"预览前 8 项 + 省略号"打印模式
2. `validate()` 直接返回 `stem_to_ext: dict[str, str]` 映射，消除 `main()` 中重复构建，减少 `split_and_copy()` 参数个数

**Blocked by:** None — 可立即开始。

**Status:** completed

- [ ] `_print_preview(label, items)` 提取后，三处 missing/empty 报告各缩减为一行调用
- [ ] `validate()` 返回值变为 `(pairs, class_counts, stem_to_ext)` 三元组
- [ ] `main()` 中不再重复遍历 `img_dir` 构建 `stem_to_ext`
- [ ] `split_and_copy()` 参数从 7 个减少到 ≤ 5 个
- [ ] 功能行为不变：dummy 数据测试校验报告和拆分结果与重构前一致
