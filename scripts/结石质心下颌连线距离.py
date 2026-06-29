"""
计算左下颌拐点到右下颌拐点的连线距离，以及结石质心在该连线上的垂足
到左、右下颌拐点的分别距离。

标记点来源：asset/病历号（体素坐标，经 affine 换算为 mm）。

几何说明：
  - lr_line_distance_mm：左、右下颌拐点间的直线距离
  - foot_to_left_mm / foot_to_right_mm：质心向 LR 连线作垂线，垂足到左/右拐点的距离
  - foot_lr_mid_offset_ratio：垂足相对 LR 中点的偏移比，
    (0.5*lr_line_distance_mm - min(foot_to_left_mm, foot_to_right_mm)) / (0.5*lr_line_distance_mm)
    0 表示垂足在中点，越接近 1 表示越靠近左或右拐点

输出 CSV：test/stone_lr_line_distance.csv
病例顺序与 asset/病历号 原始行顺序一致。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))
from extract_submandibular_roi import load_landmarks  # noqa: E402

LABELS_DIR = REPO_ROOT / "nnUNet_raw" / "Dataset001_XXY" / "labelsTr"
LANDMARK_FILE = REPO_ROOT / "asset" / "病历号"
OUTPUT_CSV = REPO_ROOT / "test" / "stone_lr_line_distance.csv"

STONE_LABEL = 1

CSV_FIELDNAMES = [
    "case_id",
    "stone_id",
    "lr_line_distance_mm",
    "foot_to_left_mm",
    "foot_to_right_mm",
    "foot_lr_mid_offset_ratio",
    "stone_voxels",
    "centroid_x",
    "centroid_y",
    "centroid_z",
]


def load_landmark_order(path: Path) -> list[str]:
    """按 asset/病历号 文件中的原始行顺序返回病历号列表。"""
    with path.open(encoding="utf-8") as f:
        return [
            row["病历号"].strip()
            for row in csv.DictReader(f, delimiter="\t")
            if row["病历号"].strip()
        ]


def voxels_to_world(affine: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """将体素坐标 (i,j,k) 批量变换为物理世界坐标 (x,y,z) mm。"""
    ijk = np.atleast_2d(ijk).astype(np.float64)
    homog = np.c_[ijk, np.ones(len(ijk))]
    return (homog @ affine.T)[:, :3]


def project_to_lr_line(
    point_w: np.ndarray, left_w: np.ndarray, right_w: np.ndarray
) -> tuple[float, float, float]:
    """将空间点投影到左-右下颌拐点连线上，返回 LR 线长及垂足到左右拐点的距离。"""
    lr_vec = right_w - left_w
    lr_len_sq = float(np.dot(lr_vec, lr_vec))
    if lr_len_sq < 1e-12:
        raise ValueError("Left and right mandible landmarks coincide")

    lr_line_distance_mm = float(np.sqrt(lr_len_sq))
    t = float(np.dot(point_w - left_w, lr_vec) / lr_len_sq)
    foot = left_w + t * lr_vec

    foot_to_left_mm = float(np.linalg.norm(foot - left_w))
    foot_to_right_mm = float(np.linalg.norm(foot - right_w))

    return lr_line_distance_mm, foot_to_left_mm, foot_to_right_mm


def foot_lr_mid_offset_ratio(
    lr_line_distance_mm: float, foot_to_left_mm: float, foot_to_right_mm: float
) -> float:
    """计算垂足相对 LR 中点的归一化偏移比。"""
    half_lr = 0.5 * lr_line_distance_mm
    if half_lr < 1e-12:
        raise ValueError("LR line length is zero")
    return (half_lr - min(foot_to_left_mm, foot_to_right_mm)) / half_lr


def analyze_case(label_path: Path) -> list[dict]:
    """分析单个病例：计算 LR 连线距离及每颗结石质心垂足到左右拐点的距离。"""
    case_id = label_path.stem.replace("image_", "").replace(".nii", "")
    left, right, _ = load_landmarks(LANDMARK_FILE, case_id)

    img = nib.load(str(label_path))
    affine = img.affine
    left_w = voxels_to_world(affine, left)[0]
    right_w = voxels_to_world(affine, right)[0]

    data = np.rint(img.get_fdata()).astype(np.uint8)
    stone_labeled, num_stones = ndimage.label(data == STONE_LABEL)
    if num_stones == 0:
        return []

    lr_line_distance_mm, _, _ = project_to_lr_line(left_w, left_w, right_w)

    rows: list[dict] = []
    for sid in range(1, num_stones + 1):
        stone_mask = stone_labeled == sid
        coords = np.argwhere(stone_mask).astype(np.float64)
        centroid = coords.mean(axis=0)
        centroid_w = voxels_to_world(affine, centroid)[0]

        _, foot_to_left_mm, foot_to_right_mm = project_to_lr_line(centroid_w, left_w, right_w)
        mid_offset = foot_lr_mid_offset_ratio(
            lr_line_distance_mm, foot_to_left_mm, foot_to_right_mm
        )

        rows.append(
            {
                "case_id": case_id,
                "stone_id": sid,
                "lr_line_distance_mm": round(lr_line_distance_mm, 3),
                "foot_to_left_mm": round(foot_to_left_mm, 3),
                "foot_to_right_mm": round(foot_to_right_mm, 3),
                "foot_lr_mid_offset_ratio": round(mid_offset, 4),
                "stone_voxels": int(stone_mask.sum()),
                "centroid_x": round(float(centroid[0]), 2),
                "centroid_y": round(float(centroid[1]), 2),
                "centroid_z": round(float(centroid[2]), 2),
            }
        )

    return rows


def main() -> None:
    """按病历号文件顺序批量分析，结果写入 stone_lr_line_distance.csv。"""
    landmark_order = load_landmark_order(LANDMARK_FILE)
    all_rows: list[dict] = []
    skipped_no_label: list[str] = []

    for case_id in landmark_order:
        label_path = LABELS_DIR / f"image_{case_id}.nii.gz"
        if not label_path.exists():
            skipped_no_label.append(case_id)
            continue
        try:
            rows = analyze_case(label_path)
            all_rows.extend(rows)
            print(f"{case_id}: {len(rows)} stone(s)")
        except Exception as exc:
            print(f"{case_id}: ERROR {exc}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nTotal stones: {len(all_rows)}")
    print(f"Saved: {OUTPUT_CSV}")
    if skipped_no_label:
        print(f"Skipped (no label file): {len(skipped_no_label)} cases")


if __name__ == "__main__":
    main()
