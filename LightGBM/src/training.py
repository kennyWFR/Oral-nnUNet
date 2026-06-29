"""LightGBM 训练与评估逻辑。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


def load_dataset(dataset_path: Path, meta_path: Path) -> tuple[pd.DataFrame, dict]:
    """加载预处理数据集与元信息。"""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"数据集不存在: {dataset_path}\n请先运行 scripts/prepare_dataset.py"
        )
    df = pd.read_csv(dataset_path, encoding="utf-8-sig")

    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        feature_cols = meta["feature_columns"]
        target_col = meta.get("target_column", "difficulty")
    else:
        target_col = "difficulty"
        exclude = {"case_id", "stone_id", "stone_count_in_image", target_col}
        feature_cols = [c for c in df.columns if c not in exclude]
        meta = {"feature_columns": feature_cols, "target_column": target_col}

    return df, meta


def prepare_xy(
    df: pd.DataFrame, feature_cols: list[str], target_col: str
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder, list[str]]:
    """提取特征矩阵 X、标签 y，并编码类别。"""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据集缺少特征列: {missing}")

    X_df = df[feature_cols].copy()
    if X_df.isna().any().any():
        na_cols = X_df.columns[X_df.isna().any()].tolist()
        print(f"警告: 以下特征含缺失值，将以列中位数填充: {na_cols}")
        X_df = X_df.fillna(X_df.median(numeric_only=True))

    le = LabelEncoder()
    y = le.fit_transform(df[target_col].astype(str))
    return X_df, y, le, list(le.classes_)


def train_lgbm_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    class_names: list[str],
    random_state: int,
) -> lgb.LGBMClassifier:
    """训练 LightGBM 多分类模型。"""
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(class_names),
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=15,
        min_child_samples=max(2, len(y) // 10),
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def evaluate_model(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict:
    """计算分类评估指标。"""
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
        ),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, labels=labels, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def cross_validate(
    X: pd.DataFrame,
    y: np.ndarray,
    class_names: list[str],
    n_splits: int,
    random_state: int,
) -> dict:
    """分层 K 折交叉验证。"""
    n_splits = min(n_splits, min(np.bincount(y)))
    if n_splits < 2:
        return {"message": "样本量或类别数过少，跳过交叉验证"}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        model = train_lgbm_classifier(X.iloc[tr_idx], y[tr_idx], class_names, random_state)
        pred = model.predict(X.iloc[va_idx])
        metrics = evaluate_model(y[va_idx], pred, class_names)
        fold_metrics.append(
            {
                "fold": fold,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
        print(
            f"  Fold {fold}: acc={metrics['accuracy']:.3f}, "
            f"macro_f1={metrics['macro_f1']:.3f}"
        )

    return {
        "n_splits": n_splits,
        "folds": fold_metrics,
        "mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "mean_macro_f1": float(np.mean([m["macro_f1"] for m in fold_metrics])),
        "mean_weighted_f1": float(np.mean([m["weighted_f1"] for m in fold_metrics])),
    }


def save_feature_importance(
    model: lgb.LGBMClassifier,
    feature_cols: list[str],
    output_path: Path,
) -> None:
    """保存特征重要性到 CSV。"""
    fi = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    fi.sort_values("importance", ascending=False).to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )


def run_training(
    df: pd.DataFrame,
    meta: dict,
    model_dir: Path,
    test_size: float = 0.1,
    n_splits: int = 5,
    random_state: int = 42,
) -> Path:
    """执行完整训练流程并保存模型，返回输出目录。"""
    feature_cols = meta["feature_columns"]
    target_col = meta.get("target_column", "difficulty")

    print(f"样本数: {len(df)}")
    print(f"特征数: {len(feature_cols)}")
    print(f"类别: {meta.get('classes', sorted(df[target_col].unique()))}")
    print(f"类别分布:\n{df[target_col].value_counts()}\n")

    if len(df) < 5:
        raise RuntimeError("样本过少，无法可靠训练。请补充难度标签 CSV。")

    X, y, label_encoder, class_names = prepare_xy(df, feature_cols, target_col)

    cv_results = None
    if n_splits > 0:
        print(f"分层 {n_splits} 折交叉验证:")
        cv_results = cross_validate(X, y, class_names, n_splits, random_state)
        if "mean_macro_f1" in cv_results:
            print(
                f"CV mean macro-F1: {cv_results['mean_macro_f1']:.3f}, "
                f"acc: {cv_results['mean_accuracy']:.3f}\n"
            )

    stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
    n_test = max(1, int(round(len(df) * test_size))) if test_size > 0 else 0
    if stratify is not None and (
        n_test < len(np.unique(y)) or (len(df) - n_test) < len(np.unique(y))
    ):
        print("警告: 样本量过小，train/test 划分不使用 stratify")
        stratify = None

    if test_size <= 0 or n_test >= len(df):
        X_train, y_train = X, y
        idx_train = np.arange(len(df))
        X_test = X.iloc[:0]
        y_test = np.array([], dtype=y.dtype)
        idx_test = np.array([], dtype=int)
        print(f"训练集: {len(X_train)}, 测试集: 0（全量训练）")
    else:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, np.arange(len(df)),
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    model = train_lgbm_classifier(X_train, y_train, class_names, random_state)

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        test_metrics = evaluate_model(y_test, y_pred, class_names)
        print("\n测试集评估:")
        print(test_metrics["classification_report"])
        print("混淆矩阵:")
        print(np.array(test_metrics["confusion_matrix"]))
    else:
        test_metrics = {"accuracy": None, "macro_f1": None, "weighted_f1": None, "confusion_matrix": []}
        print("\n无 hold-out 测试集，跳过测试集评估")

    y_train_pred = label_encoder.inverse_transform(model.predict(X_train).astype(int))
    train_case_ids = df.iloc[idx_train]["case_id"].astype(str).tolist()
    train_true = df.iloc[idx_train][target_col].astype(str).tolist()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = model_dir / f"lgbm_difficulty_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(label_encoder, out_dir / "label_encoder.joblib")
    with (out_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    save_feature_importance(model, feature_cols, out_dir / "feature_importance.csv")

    train_pred_df = pd.DataFrame(
        {
            "case_id": train_case_ids,
            "true_difficulty": train_true,
            "pred_difficulty": list(y_train_pred),
        }
    )
    train_pred_df.to_csv(out_dir / "train_predictions.csv", index=False, encoding="utf-8-sig")

    print("\n训练集推理:")
    for cid, true, pred in zip(train_case_ids, train_true, y_train_pred):
        mark = "OK" if str(true) == str(pred) else "X"
        print(f"  {cid}: true={true}, pred={pred} [{mark}]")

    report = {
        "run_id": run_id,
        "test_size": test_size,
        "n_samples": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "classes": class_names,
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "classification_report"},
        "test_classification_report": test_metrics.get("classification_report"),
        "cv_results": cv_results,
        "train_case_ids": train_case_ids,
        "test_case_ids": [str(x) for x in df.iloc[idx_test]["case_id"].tolist()],
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return out_dir
