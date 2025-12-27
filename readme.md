中文 | [English](readme_en.md)

# Branch and Price Solver Demo (VRPTW)

本项目是一个基于 Python 实现的 **Branch and Price (分支定价)** 算法demo，主要用于解决带时间窗的车辆路径问题 (VRPTW)。

## 📌 项目定位

本项目旨在展示 Branch and Price 算法的核心实现逻辑，包括列生成 (Column Generation) 与分支定界 (Branch and Bound) 的结合。

## 🛠️ 环境要求

* **Python 版本:** 3.10
* **求解器:** [CBC](https://github.com/coin-or/Cbc) (需要预先自行安装)
* **建模框架:** `python-mip`

## 🚀 快速开始

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **运行 Demo**
```bash
python bp_example.py
```

## 📊 算例支持

项目内置了以下国际标准测试集，位于 `./instance` 目录下：

* **Solomon 算例:** 包含 R、C、RC 三类经典测试集。
* **Homberger 算例:** 针对 200-1000 节点的大规模场景扩展算例。

## 📂 核心代码结构

* `src/solver/branch_and_price`: 核心算法逻辑实现。
* `bp_example.py`: 算法入口与配置示例。

## ⚠️ 特别说明 (Notes)

1. **纯 Demo 性质:** 本项目仅作为算法逻辑展示，旨在分享实现思路，不建议直接用于生产环境。
2. **停止维护:** **本项目目前已停止更新，后续无计划进行功能扩展或维护。** 欢迎 Fork 并在其基础上自行探索。
