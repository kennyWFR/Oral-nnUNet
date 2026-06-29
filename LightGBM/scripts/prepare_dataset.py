"""
结石难度分类 — 数据预处理入口。

用法（在项目根目录执行）：
  python LightGBM/scripts/prepare_dataset.py
  python LightGBM/scripts/prepare_dataset.py --labels-csv LightGBM/data/labels/my_labels.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LIGHTGBM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIGHTGBM_ROOT))

from config.paths import (  # noqa: E402
    DATASET_CSV,
    DATASET_META_JSON,
    DIFFICULTY_SCORES_FILE,
    FEATURES_CSV,
    LABELS_CSV,
    PROCESSED_DIR,
)
from src.preprocessing import build_dataset  # noqa: E402


def main() -> None:
    """命令行入口：预处理并保存训练数据集。"""
    parser = argparse.ArgumentParser(description="结石难度分类数据预处理")
    parser.add_argument("--features-csv", type=Path, default=FEATURES_CSV)
    parser.add_argument("--labels-csv", type=Path, default=LABELS_CSV,
                        help="难度评分文件，默认 asset/难度评分")
    parser.add_argument("--output-csv", type=Path, default=DATASET_CSV)
    parser.add_argument("--meta-json", type=Path, default=DATASET_META_JSON)
    parser.add_argument("--case-col", default="case_id", help="难度 CSV 中的病历号列名")
    parser.add_argument("--label-col", default="difficulty", help="难度 CSV 中的标签列名")
    args = parser.parse_args()

    if not args.features_csv.exists():
        print(f"错误: 特征文件不存在 - {args.features_csv}")
        print("请先运行: python scripts/结石综合特征.py")
        sys.exit(1)
    if not args.labels_csv.exists():
        print(f"错误: 难度评分文件不存在 - {args.labels_csv}")
        print(f"默认路径: {DIFFICULTY_SCORES_FILE}")
        sys.exit(1)

    dataset, meta = build_dataset(
        args.features_csv,
        args.labels_csv,
        case_col=args.case_col,
        label_col=args.label_col,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    with args.meta_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"预处理完成: {len(dataset)} 条样本")
    print(f"类别分布: {meta['class_counts']}")
    print(f"数据集: {args.output_csv}")
    print(f"元信息: {args.meta_json}")


if __name__ == "__main__":
    main()
