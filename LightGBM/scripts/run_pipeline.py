"""
结石难度分类 — 一键流水线（预处理 + 训练）。

用法：
  python LightGBM/scripts/run_pipeline.py
  python LightGBM/scripts/run_pipeline.py --n-splits 0
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def main() -> None:
    """依次执行预处理与训练。"""
    steps = [
        [PYTHON, str(SCRIPTS_DIR / "prepare_dataset.py"), *sys.argv[1:]],
        [PYTHON, str(SCRIPTS_DIR / "train_lgbm.py"), *sys.argv[1:]],
    ]
    for cmd in steps:
        print("\n>>>", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
