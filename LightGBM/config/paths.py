"""LightGBM 模块路径配置。"""
from pathlib import Path

# LightGBM/ 目录
LIGHTGBM_ROOT = Path(__file__).resolve().parents[1]
# nnUNet 项目根目录
REPO_ROOT = LIGHTGBM_ROOT.parent

# 外部输入（由 scripts/结石综合特征.py 生成）
FEATURES_CSV = REPO_ROOT / "test" / "stone_features_combined.csv"

# 难度评分（单结石样本，asset/难度评分：每行「case_id 分数」）
DIFFICULTY_SCORES_FILE = REPO_ROOT / "asset" / "难度评分"

# 数据目录
DATA_DIR = LIGHTGBM_ROOT / "data"
LABELS_DIR = DATA_DIR / "labels"
PROCESSED_DIR = DATA_DIR / "processed"

# 兼容旧路径命名
LABELS_CSV = DIFFICULTY_SCORES_FILE
LABELS_TEMPLATE_CSV = LABELS_DIR / "stone_difficulty_labels.template.csv"

DATASET_CSV = PROCESSED_DIR / "stone_difficulty_dataset.csv"
DATASET_META_JSON = PROCESSED_DIR / "dataset_meta.json"

# 模型与训练输出
MODELS_DIR = LIGHTGBM_ROOT / "models"

# 预处理常量
ID_COLUMNS = frozenset({"case_id", "stone_id"})
DROP_COLUMNS = frozenset({"stone_count_in_image"})
TARGET_COLUMN = "difficulty"
