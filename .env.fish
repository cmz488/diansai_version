# conda 项目配置 — 在 yolo 项目目录下 source 此文件
set -gx CONDARC (dirname (realpath (status filename)))/.condarc
