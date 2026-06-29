"""
分析每个结石与颌下腺的关系（outer / intersect / inner），并计算结石取出距离。

输出 CSV：test/stone_extraction_distance.csv

=== 关系判定（基于结石外表面体素到腺体的 EDT 距离，阈值 ADJACENT_VOXELS=2）===

对每个结石连通域 × 每个腺体连通域：
  outer     : 所有外表面点到该腺体最近距离 > 2 体素（min_d > 2）
  inner     : 每个外表面点到该腺体最近距离均 ≤ 2 体素（max_d ≤ 2，被腺体完全包裹）
  intersect : 存在外表面点 ≤ 2，但并非全部 ≤ 2（min_d ≤ 2 且 max_d > 2）

多腺体汇总（优先级 inner > intersect > outer）：
  任一腺体为 inner   → 整体 inner
  否则任一 intersect → 整体 intersect
  否则               → 整体 outer

=== 结石取出距离 extraction_distance_mm ===

  outer     : -1（待定）
  intersect : 0
  inner     : 结石质心到「对应 inner 腺体」的内侧+前侧+上侧表面体素的最短物理距离(mm)

=== CSV 字段说明（CSV_FIELD_DOCS）===
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "test"))
from extract_submandibular_roi import (  # noqa: E402
    build_local_axes,
    extract_surface_mask,
    load_landmarks,
    select_med_ant_sup_surface_voxels,
)

DATASET_DIR = PROJECT / "nnUNet_raw" / "Dataset001_XXY"
LABELS_DIR = DATASET_DIR / "labelsTr"
LANDMARK_FILE = PROJECT / "asset" / "病历号"
OUTPUT_CSV = PROJECT / "test" / "stone_extraction_distance.csv"

STONE_LABEL = 1  # dataset.json: stone
GLAND_LABEL = 2  # dataset.json: bland（颌下腺）
OUTER_DISTANCE = -1.0
ADJACENT_VOXELS = 2  # 外表面点到腺体距离阈值（体素）

# CSV 各列含义（写入脚本便于查阅；首行注释不写入 CSV 文件）
CSV_FIELD_DOCS: dict[str, str] = {
    "case_id": "病历号，与 asset/病历号 及 labelsTr 文件名 image_{case_id}.nii.gz 对应",
    "stone_id": "该病例内结石连通域编号，从 1 起（label=1 的独立连通块）",
    "relation": "结石与腺体的汇总关系：outer / intersect / inner（见模块说明）",
    "extraction_distance_mm": "结石取出距离(mm)：outer=-1，intersect=0，inner=质心到 MAS 表面最短距离",
    "min_surface_dist_voxels": "结石外表面点到「最近腺体」的最短 EDT 距离（体素，诊断用）",
    "max_surface_dist_voxels": "结石外表面点到「最近腺体」的最远 EDT 距离（体素；inner 时 ≤2）",
    "stone_voxels": "该结石连通域体素总数",
    "centroid_x": "结石质心 x（体素坐标，与 NIfTI 数组下标一致）",
    "centroid_y": "结石质心 y（体素坐标）",
    "centroid_z": "结石质心 z（体素坐标）",
}
CSV_FIELDNAMES: list[str] = list(CSV_FIELD_DOCS.keys())


def case_id_from_label_path(path: Path) -> str:
    """从 labelsTr 文件名 image_{case_id}.nii.gz 中解析病历号。"""
    m = re.match(r"image_(.+)\.nii\.gz$", path.name)
    if not m:
        raise ValueError(f"Unrecognized label filename: {path.name}")
    return m.group(1)


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


def min_distance_mm(centroid_ijk: np.ndarray, target_ijks: np.ndarray, affine: np.ndarray) -> float:
    """计算质心到目标体素集合的最短欧氏距离（mm）。"""
    if len(target_ijks) == 0:
        return float("nan")
    c_world = voxels_to_world(affine, centroid_ijk.reshape(1, -1))[0]
    t_world = voxels_to_world(affine, target_ijks)
    return float(cKDTree(t_world).query(c_world)[0])


def stone_surface_distances_to_gland(
    stone_mask: np.ndarray, gland_mask: np.ndarray
) -> tuple[float, float]:
    """计算结石外表面各体素到腺体的 EDT，返回 (最小距离, 最大距离)（体素单位）。"""
    surface = extract_surface_mask(stone_mask)
    coords = np.argwhere(surface)
    if len(coords) == 0:
        coords = np.argwhere(stone_mask)
    if len(coords) == 0:
        raise ValueError("Empty stone mask")

    dt = ndimage.distance_transform_edt(~gland_mask)
    dists = dt[coords[:, 0], coords[:, 1], coords[:, 2]]
    return float(dists.min()), float(dists.max())


def stone_gland_relation(stone_mask: np.ndarray, gland_mask: np.ndarray) -> str:
    """根据外表面 EDT 判定单颗结石与单个腺体的关系：outer / intersect / inner。"""
    min_d, max_d = stone_surface_distances_to_gland(stone_mask, gland_mask)

    if min_d > ADJACENT_VOXELS:
        return "outer"
    if max_d <= ADJACENT_VOXELS:
        return "inner"
    return "intersect"


def aggregate_relation(relations: list[str]) -> str:
    """汇总多腺体关系，优先级：inner > intersect > outer。"""
    if "inner" in relations:
        return "inner"
    if "intersect" in relations:
        return "intersect"
    return "outer"


def compute_mas_masks(
    gland_labeled: np.ndarray,
    num_glands: int,
    left: np.ndarray,
    right: np.ndarray,
    chin: np.ndarray,
) -> dict[int, np.ndarray]:
    """为每个腺体连通域计算内侧+前侧+上侧（MAS）表面体素 mask，用于 inner 取出距离。"""
    mas_by_gland: dict[int, np.ndarray] = {}
    for gid in range(1, num_glands + 1):
        cc = gland_labeled == gid
        coords = np.argwhere(cc).astype(np.float64)
        centroid = coords.mean(axis=0)
        origin, e_lr, e_ap, e_s = build_local_axes(left, right, chin, centroid)
        mas_mask, _ = select_med_ant_sup_surface_voxels(cc, origin, e_lr, e_ap, e_s)
        mas_by_gland[gid] = mas_mask
    return mas_by_gland


def analyze_case(label_path: Path) -> list[dict]:
    """分析单个病例：判定每颗结石与腺体的关系，并计算结石取出距离。"""
    case_id = case_id_from_label_path(label_path)
    left, right, chin = load_landmarks(LANDMARK_FILE, case_id)

    img = nib.load(str(label_path))
    affine = img.affine
    data = np.rint(img.get_fdata()).astype(np.uint8)

    stone_labeled, num_stones = ndimage.label(data == STONE_LABEL)
    gland_labeled, num_glands = ndimage.label(data == GLAND_LABEL)
    if num_stones == 0:
        return []
    if num_glands == 0:
        raise RuntimeError(f"{case_id}: no gland (label={GLAND_LABEL}) found")

    mas_by_gland = compute_mas_masks(gland_labeled, num_glands, left, right, chin)
    rows: list[dict] = []

    for sid in range(1, num_stones + 1):
        stone_mask = stone_labeled == sid
        stone_coords = np.argwhere(stone_mask).astype(np.float64)
        centroid = stone_coords.mean(axis=0)

        per_gland = {
            gid: stone_gland_relation(stone_mask, gland_labeled == gid)
            for gid in range(1, num_glands + 1)
        }
        relation = aggregate_relation(list(per_gland.values()))

        closest_min = float("inf")
        closest_max = float("inf")
        for gid in range(1, num_glands + 1):
            mn, mx = stone_surface_distances_to_gland(stone_mask, gland_labeled == gid)
            if mn < closest_min:
                closest_min, closest_max = mn, mx

        if relation == "outer":
            distance = OUTER_DISTANCE
        elif relation == "intersect":
            distance = 0.0
        else:
            inner_ids = [gid for gid, rel in per_gland.items() if rel == "inner"]
            mas_points = []
            for gid in inner_ids:
                mas_ijk = np.argwhere(mas_by_gland[gid])
                if len(mas_ijk) > 0:
                    mas_points.append(mas_ijk)
            if not mas_points:
                distance = float("nan")
            else:
                all_mas = np.vstack(mas_points)
                distance = min_distance_mm(centroid, all_mas, affine)

        rows.append(
            {
                "case_id": case_id,
                "stone_id": sid,
                "relation": relation,
                "extraction_distance_mm": distance,
                "min_surface_dist_voxels": closest_min,
                "max_surface_dist_voxels": closest_max,
                "stone_voxels": int(stone_mask.sum()),
                "centroid_x": round(float(centroid[0]), 2),
                "centroid_y": round(float(centroid[1]), 2),
                "centroid_z": round(float(centroid[2]), 2),
            }
        )

    return rows


def main() -> None:
    """按病历号文件顺序批量分析，结果写入 stone_extraction_distance.csv。"""
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

    print(f"\nTotal stones analyzed: {len(all_rows)}")
    print(f"Saved: {OUTPUT_CSV}")
    if skipped_no_label:
        print(f"Skipped (no label file): {len(skipped_no_label)} cases")


if __name__ == "__main__":
    main()
