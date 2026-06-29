"""结石难度数据集预处理逻辑。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.paths import DROP_COLUMNS, ID_COLUMNS, TARGET_COLUMN


def normalize_case_id(raw: object) -> str:
    """规范化病历号，便于与不同来源 CSV 对齐。"""
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    s = s.replace("\\", "/")
    if "/" in s:
        s = Path(s).name
    for prefix in ("image_", "Image_", "IMAGE_"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    for suffix in (".nii.gz", ".nii"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s.strip()


def load_features(path: Path) -> pd.DataFrame:
    """加载综合特征 CSV，规范化 case_id 并筛选单结石样本。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "case_id" not in df.columns:
        raise ValueError(f"特征 CSV 缺少 case_id 列: {path}")

    df["case_id"] = df["case_id"].map(normalize_case_id)
    df = df[df["case_id"] != ""].copy()

    if "stone_count_in_image" not in df.columns:
        raise ValueError("特征 CSV 缺少 stone_count_in_image 列")

    df["stone_count_in_image"] = pd.to_numeric(df["stone_count_in_image"], errors="coerce")
    all_count = len(df)
    single = df[df["stone_count_in_image"] == 1].copy()
    multi = df[df["stone_count_in_image"] > 1]

    print(f"特征 CSV: 共 {all_count} 行")
    print(f"  单结石样本: {len(single)} 行")
    print(f"  多结石样本: {len(multi)} 行（难度标签通常不包含，已排除）")

    dup = single["case_id"].duplicated(keep=False)
    if dup.any():
        dup_ids = sorted(single.loc[dup, "case_id"].unique())
        raise ValueError(f"单结石样本中存在重复 case_id: {dup_ids}")

    return single


def load_labels(path: Path, case_col: str, label_col: str) -> pd.DataFrame:
    """加载难度标签；支持 CSV/TSV 或 asset/难度评分 的空格分隔格式。"""
    if not path.exists():
        raise FileNotFoundError(f"难度评分文件不存在: {path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"难度评分文件为空: {path}")

    first_line = raw.splitlines()[0]
    if "," in first_line or "\t" in first_line:
        df = pd.read_csv(path, encoding="utf-8-sig")
        if case_col not in df.columns:
            raise ValueError(f"标签 CSV 缺少列 {case_col!r}，现有列: {list(df.columns)}")
        if label_col not in df.columns:
            raise ValueError(f"标签 CSV 缺少列 {label_col!r}，现有列: {list(df.columns)}")
        out = df[[case_col, label_col]].copy()
        out.columns = ["case_id", TARGET_COLUMN]
    else:
        rows: list[tuple[str, str]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"无法解析难度评分行: {line!r}")
            rows.append((parts[0], parts[1]))
        out = pd.DataFrame(rows, columns=["case_id", TARGET_COLUMN])

    out["case_id"] = out["case_id"].map(normalize_case_id)
    out[TARGET_COLUMN] = out[TARGET_COLUMN].astype(str).str.strip()
    out = out[(out["case_id"] != "") & (out[TARGET_COLUMN] != "")].copy()

    dup = out["case_id"].duplicated(keep=False)
    if dup.any():
        dup_ids = sorted(out.loc[dup, "case_id"].unique())
        raise ValueError(f"难度 CSV 中存在重复 case_id: {dup_ids}")

    return out


def encode_relation(df: pd.DataFrame) -> pd.DataFrame:
    """将 relation 列 one-hot 编码为 relation_inner / relation_intersect / relation_outer。"""
    if "relation" not in df.columns:
        return df
    dummies = pd.get_dummies(df["relation"], prefix="relation", dtype=int)
    expected = ["relation_inner", "relation_intersect", "relation_outer"]
    for col in expected:
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[expected]
    return pd.concat([df.drop(columns=["relation"]), dummies], axis=1)


def coerce_numeric_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """将特征列转为数值；空字符串转为 NaN。"""
    out = df.copy()
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_dataset(
    features_path: Path,
    labels_path: Path,
    case_col: str = "case_id",
    label_col: str = "difficulty",
) -> tuple[pd.DataFrame, dict]:
    """合并特征与标签，返回训练数据集及元信息。"""
    feat_single = load_features(features_path)
    labels = load_labels(labels_path, case_col, label_col)

    merged = feat_single.merge(labels, on="case_id", how="inner")
    feat_only = set(feat_single["case_id"]) - set(labels["case_id"])
    label_only = set(labels["case_id"]) - set(feat_single["case_id"])

    print("\n标签对齐:")
    print(f"  难度 CSV 标签数: {len(labels)}")
    print(f"  成功匹配: {len(merged)}")
    print(f"  有特征无标签: {len(feat_only)}")
    print(f"  有标签无特征(或属多结石): {len(label_only)}")

    if feat_only:
        print(f"  有特征无标签 case_id 示例: {sorted(feat_only)[:10]}")
    if label_only:
        print(f"  有标签无匹配特征 case_id 示例: {sorted(label_only)[:10]}")

    if merged.empty:
        raise RuntimeError(
            "合并后无样本。请检查 case_id 是否一致，以及难度 CSV 是否仅包含单结石病例。"
        )

    merged = encode_relation(merged)

    exclude = set(ID_COLUMNS) | set(DROP_COLUMNS) | {TARGET_COLUMN}
    feature_cols = [c for c in merged.columns if c not in exclude]

    merged = coerce_numeric_features(merged, feature_cols)

    na_counts = merged[feature_cols].isna().sum()
    na_cols = na_counts[na_counts > 0]
    if not na_cols.empty:
        print("\n含缺失值的特征列:")
        for col, cnt in na_cols.items():
            print(f"  {col}: {cnt}")

    classes = sorted(merged[TARGET_COLUMN].unique())
    meta = {
        "features_csv": str(features_path),
        "labels_csv": str(labels_path),
        "target_column": TARGET_COLUMN,
        "classes": classes,
        "feature_columns": feature_cols,
        "id_columns": sorted(ID_COLUMNS),
        "n_samples": len(merged),
        "n_single_stone_features": len(feat_single),
        "n_labels": len(labels),
        "unmatched_feature_case_ids": sorted(feat_only),
        "unmatched_label_case_ids": sorted(label_only),
        "class_counts": merged[TARGET_COLUMN].value_counts().to_dict(),
    }
    return merged, meta
