# 01 — conda 环境搭建 + 数据集拆分脚本

**What to build:** 用户能在 RTX 5060 训练服务器上一键创建 `yolo` conda 环境（Python 3.10 + PyTorch 2.4 + CUDA 12.4 + Ultralytics），验证 GPU 可用。同时提供数据集拆分脚本：读取原始标注图片，校验缺漏，统计各类别框数，按 7:2:1 拆分 train/val/test，生成 `data.yaml`。

**Blocked by:** None — 可立即开始。

**Status:** completed

- [ ] `environment.yml` 文件存在，`conda env create -f environment.yml` 成功创建名为 `yolo` 的环境
- [ ] `conda activate yolo && python -c "import torch; print(torch.cuda.is_available())"` 输出 `True`
- [ ] `scripts/split_dataset.py --raw yolo/raw --out yolo --ratio 7:2:1` 能正确校验并拆分数据集
- [ ] 校验报告包含：图片/标注配对数量、缺标注列表、缺图片列表、各类别框数统计
- [ ] 拆分后在 train/val/test 下各有 images/ 和 labels/ 子目录
- [ ] 自动生成的 data.yaml 包含正确的 path、train、val、test、nc、names 字段
