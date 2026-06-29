"""
结石难度分类 — LightGBM 训练入口。

用法（在项目根目录执行）：
  python LightGBM/scripts/prepare_dataset.py
  python LightGBM/scripts/train_lgbm.py
  python LightGBM/scripts/train_lgbm.py --n-splits 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LIGHTGBM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIGHTGBM_ROOT))

from config.paths import DATASET_CSV, DATASET_META_JSON, MODELS_DIR  # noqa: E402
from src.training import load_dataset, run_training  # noqa: E402


def main() -> None:
    """命令行入口：训练 LightGBM 并保存模型与评估结果。"""
    parser = argparse.ArgumentParser(description="结石难度 LightGBM 分类训练")
    parser.add_argument("--dataset-csv", type=Path, default=DATASET_CSV)
    parser.add_argument("--meta-json", type=Path, default=DATASET_META_JSON)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--test-size", type=float, default=0.1, help="hold-out 测试集比例")
    parser.add_argument("--n-splits", type=int, default=5, help="交叉验证折数，0 表示跳过")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    try:
        df, meta = load_dataset(args.dataset_csv, args.meta_json)
        out_dir = run_training(
            df,
            meta,
            model_dir=args.model_dir,
            test_size=args.test_size,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"错误: {exc}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"模型已保存: {out_dir}")
    print("  model.joblib")
    print("  label_encoder.joblib")
    print("  feature_importance.csv")
    print("  metrics.json")


if __name__ == "__main__":
    main()
