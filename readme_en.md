[中文](readme.md) | English


# Branch and Price Solver Demo (VRPTW)

This project is a Python-based demonstration of the **Branch and Price** algorithm, specifically designed to solve the Vehicle Routing Problem with Time Windows (VRPTW).

## 📌 Project Purpose

The primary goal of this project is to showcase the core implementation logic of the Branch and Price algorithm, including the integration of **Column Generation** and **Branch and Bound**.

## 🛠️ Requirements

* **Python Version:** 3.10
* **Solver:** [CBC](https://github.com/coin-or/Cbc) (Must be installed separately)
* **Modeling Framework:** `python-mip`

## 🚀 Quick Start

1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

2. **Run Demo**

```bash
python bp_example.py
```

## 📊 Benchmark Support

The project includes standard international test instances located in the `./instance` directory:

* **Solomon Instances:** Classic test sets including Type R (Random), Type C (Clustered), and Type RC (Mixed).
* **Homberger Instances:** Extended benchmarks for large-scale scenarios (200-1000 customer nodes).

## 📂 Core Structure

* `src/solver/branch_and_price`: Implementation of the core algorithmic logic.
* `bp_example.py`: Entry point for the algorithm and configuration examples.

## ⚠️ Notes

1. **Demo Only:** This project is intended solely as a demonstration of algorithmic logic and implementation ideas. It is not recommended for production environments.
2. **Maintenance Status:** **This project is no longer actively maintained.** There are no plans for future functional updates or maintenance. Feel free to Fork the repository and explore the code on your own.
