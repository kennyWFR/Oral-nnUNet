"""
计算每颗结石质心到「局部水平面」的垂直距离。

局部水平面：由左下颌角、右下颌角、颏点（asset/病历号）三点确定的平面。
垂直距离：质心到该平面的有符号垂直距离（mm）。
  正值 = 质心位于平面下方
  负值 = 质心位于平面上方（朝颅侧）

输出 CSV：test/stone_vertical_distance.csv
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
from extract_submandibular_roi import build_local_axes, load_landmarks  # noqa: E402

LABELS_DIR = REPO_ROOT / "nnUNet_raw" / "Dataset001_XXY" / "labelsTr"
LANDMARK_FILE = REPO_ROOT / "asset" / "病历号"
OUTPUT_CSV = REPO_ROOT / "test" / "stone_vertical_distance.csv"

STONE_LABEL = 1

CSV_FIELDNAMES = [
    "case_id",
    "stone_id",
    "vertical_distance_mm",
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


def ijk_direction_to_world(affine: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """将体素空间方向向量经 affine 旋转部分变换为世界坐标下的单位方向向量。"""
    vec = affine[:3, :3] @ np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Zero-length direction vector after affine transform")
    return vec / norm


def local_horizontal_plane_normal_world(
    affine: np.ndarray,
    left_ijk: np.ndarray,
    right_ijk: np.ndarray,
    chin_ijk: np.ndarray,
    reference_ijk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """由左/右下颌角与颏点确定局部水平面，返回 (平面上一点, 朝上单位法向) 的世界坐标。"""
    left_w = voxels_to_world(affine, left_ijk)[0]
    right_w = voxels_to_world(affine, right_ijk)[0]
    chin_w = voxels_to_world(affine, chin_ijk)[0]

    normal = np.cross(right_w - left_w, chin_w - left_w)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        raise ValueError("Landmarks are nearly collinear; horizontal plane is undefined")
    normal = normal / norm

    _, _, _, e_s = build_local_axes(left_ijk, right_ijk, chin_ijk, reference_ijk)
    e_s_world = ijk_direction_to_world(affine, e_s)
    if np.dot(normal, e_s_world) < 0:
        normal = -normal

    plane_point = left_w
    return plane_point, normal


def signed_vertical_distance_mm(
    affine: np.ndarray,
    point_ijk: np.ndarray,
    left_ijk: np.ndarray,
    right_ijk: np.ndarray,
    chin_ijk: np.ndarray,
    reference_ijk: np.ndarray,
) -> float:
    """计算空间点到局部水平面的有符号垂直距离（mm）；负值表示位于平面上方（颅侧）。"""
    plane_point, normal = local_horizontal_plane_normal_world(
        affine, left_ijk, right_ijk, chin_ijk, reference_ijk
    )
    point_w = voxels_to_world(affine, point_ijk)[0]
    return -float(np.dot(point_w - plane_point, normal))


def analyze_case(label_path: Path) -> list[dict]:
    """分析单个病例：计算每颗结石质心到局部水平面的垂直距离。"""
    case_id = label_path.stem.replace("image_", "").replace(".nii", "")
    left, right, chin = load_landmarks(LANDMARK_FILE, case_id)

    img = nib.load(str(label_path))
    affine = img.affine
    data = np.rint(img.get_fdata()).astype(np.uint8)

    stone_labeled, num_stones = ndimage.label(data == STONE_LABEL)
    if num_stones == 0:
        return []

    rows: list[dict] = []
    for sid in range(1, num_stones + 1):
        stone_mask = stone_labeled == sid
        coords = np.argwhere(stone_mask).astype(np.float64)
        centroid = coords.mean(axis=0)

        dist_mm = signed_vertical_distance_mm(
            affine, centroid, left, right, chin, reference_ijk=centroid
        )

        rows.append(
            {
                "case_id": case_id,
                "stone_id": sid,
                "vertical_distance_mm": round(dist_mm, 3),
                "stone_voxels": int(stone_mask.sum()),
                "centroid_x": round(float(centroid[0]), 2),
                "centroid_y": round(float(centroid[1]), 2),
                "centroid_z": round(float(centroid[2]), 2),
            }
        )

    return rows


def main() -> None:
    """按病历号文件顺序批量分析，结果写入 stone_vertical_distance.csv。"""
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
