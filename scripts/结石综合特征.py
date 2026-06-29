"""
结石综合特征提取流水线。

步骤：
  1. 调用 清理腺体连通域.py 清洗 labelsTr（默认输出到 labelsTr_cleaned，不覆盖原始数据）
  2. 在清洗后的标签上，合并以下四类特征（每颗结石一行）：
     - 结石几何特征（体积、表面积、形态等）
     - 结石取出距离（与腺体关系、extraction_distance_mm）
     - 质心到局部水平面垂直距离
     - 质心相对 LR 下颌连线的位置特征
  3. 输出综合 CSV，供机器学习训练使用

默认仅处理 asset/病历号 中有 landmark 且存在标签文件的病例，顺序与病历号一致。

用法：
  python scripts/结石综合特征.py
  python scripts/结石综合特征.py --skip-clean --labels-dir nnUNet_raw/Dataset001_XXY/labelsTr
  python scripts/结石综合特征.py --output-csv test/my_features.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT / "test"))

from extract_submandibular_roi import load_landmarks  # noqa: E402

DATASET_DIR = REPO_ROOT / "nnUNet_raw" / "Dataset001_XXY"
DEFAULT_LABELS_DIR = DATASET_DIR / "labelsTr"
DEFAULT_CLEANED_DIR = DATASET_DIR / "labelsTr_cleaned"
LANDMARK_FILE = REPO_ROOT / "asset" / "病历号"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "test" / "stone_features_combined.csv"

STONE_LABEL = 1
GLAND_LABEL = 2
OUTER_DISTANCE = -1.0
ADJACENT_VOXELS = 2
MIN_GLAND_VOXELS = 10000

CSV_FIELDNAMES = [
    "case_id",
    "stone_id",
    "stone_count_in_image",
    "volume_voxels",
    "volume_mm3",
    "surface_area_mm2",
    "equivalent_diameter_mm",
    "max_feret_diameter_mm",
    "major_axis_length_mm",
    "minor_axis_length_mm",
    "axis_ratio_major_minor",
    "bbox_length_x_mm",
    "bbox_length_y_mm",
    "bbox_length_z_mm",
    "bbox_diagonal_mm",
    "sphericity",
    "compactness",
    "centroid_x_mm",
    "centroid_y_mm",
    "centroid_z_mm",
    "relation",
    "extraction_distance_mm",
    "min_surface_dist_voxels",
    "max_surface_dist_voxels",
    "vertical_distance_mm",
    "lr_line_distance_mm",
    "foot_to_left_mm",
    "foot_to_right_mm",
    "foot_lr_mid_offset_ratio",
    "centroid_x",
    "centroid_y",
    "centroid_z",
]


def _import_script(module_name: str, filename: str):
    """按文件路径动态加载 scripts 目录下的 Python 模块。"""
    path = SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_clean_mod = _import_script("clean_gland", "清理腺体连通域.py")
_geom_mod = _import_script("stone_geometry", "结石几何特征.py")
_ext_mod = _import_script("stone_extraction", "结石取出距离.py")
_lr_mod = _import_script("stone_lr", "结石质心下颌连线距离.py")
_vert_mod = _import_script("stone_vertical", "结石质心垂直距离.py")


def load_landmark_order(path: Path) -> list[str]:
    """按 asset/病历号 文件中的原始行顺序返回病历号列表。"""
    with path.open(encoding="utf-8") as f:
        return [
            row["病历号"].strip()
            for row in csv.DictReader(f, delimiter="\t")
            if row["病历号"].strip()
        ]


def run_gland_cleaning(
    source_dir: Path,
    output_dir: Path,
    min_voxels: int = MIN_GLAND_VOXELS,
) -> None:
    """将 source_dir 中标签清洗后写入 output_dir（去除腺体小连通域）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    label_files = sorted(source_dir.glob("*.nii.gz"))
    if not label_files:
        raise FileNotFoundError(f"未找到标签文件: {source_dir}")

    print(f"清洗腺体连通域: {source_dir} -> {output_dir}")
    print(f"阈值: 保留体素数 > {min_voxels} 的腺体连通域")
    print("=" * 70)

    total_removed = 0
    for label_path in label_files:
        stats = _clean_mod.process_file(
            label_path,
            output_dir / label_path.name,
            min_voxels=min_voxels,
            dry_run=False,
        )
        total_removed += stats["removed_count"]
        print(
            f"{stats['file']}: 连通域 {stats['component_count']} -> "
            f"保留 {stats['kept_count']}, 删除 {stats['removed_count']} 个"
        )

    print("=" * 70)
    print(f"清洗完成: {len(label_files)} 个文件, 共删除小连通域 {total_removed} 个\n")


def analyze_case(label_path: Path) -> list[dict]:
    """对单个病例提取全部结石综合特征（一行一颗结石）。"""
    case_id = label_path.name.replace("image_", "").replace(".nii.gz", "")
    left, right, chin = load_landmarks(LANDMARK_FILE, case_id)

    img = nib.load(str(label_path))
    affine = img.affine
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    data = np.rint(img.get_fdata()).astype(np.int16)
    data, _ = _clean_mod.clean_gland_labels(data, min_voxels=MIN_GLAND_VOXELS)
    data = data.astype(np.uint8)

    stone_labeled, num_stones = ndimage.label(data == STONE_LABEL)
    if num_stones == 0:
        return []

    gland_labeled, num_glands = ndimage.label(data == GLAND_LABEL)
    if num_glands == 0:
        raise RuntimeError(f"{case_id}: 清洗后未找到腺体 (label={GLAND_LABEL})")

    mas_by_gland = _ext_mod.compute_mas_masks(gland_labeled, num_glands, left, right, chin)
    left_w = _lr_mod.voxels_to_world(affine, left)[0]
    right_w = _lr_mod.voxels_to_world(affine, right)[0]
    lr_line_distance_mm, _, _ = _lr_mod.project_to_lr_line(left_w, left_w, right_w)

    rows: list[dict] = []
    for sid in range(1, num_stones + 1):
        stone_mask = stone_labeled == sid
        stone_coords = np.argwhere(stone_mask).astype(np.float64)
        centroid = stone_coords.mean(axis=0)
        centroid_w = _lr_mod.voxels_to_world(affine, centroid)[0]

        geom = _geom_mod.compute_stone_features(stone_mask, spacing)

        per_gland = {
            gid: _ext_mod.stone_gland_relation(stone_mask, gland_labeled == gid)
            for gid in range(1, num_glands + 1)
        }
        relation = _ext_mod.aggregate_relation(list(per_gland.values()))

        closest_min = float("inf")
        closest_max = float("inf")
        for gid in range(1, num_glands + 1):
            mn, mx = _ext_mod.stone_surface_distances_to_gland(stone_mask, gland_labeled == gid)
            if mn < closest_min:
                closest_min, closest_max = mn, mx

        if relation == "outer":
            extraction_distance = OUTER_DISTANCE
        elif relation == "intersect":
            extraction_distance = 0.0
        else:
            inner_ids = [gid for gid, rel in per_gland.items() if rel == "inner"]
            mas_points = []
            for gid in inner_ids:
                mas_ijk = np.argwhere(mas_by_gland[gid])
                if len(mas_ijk) > 0:
                    mas_points.append(mas_ijk)
            extraction_distance = (
                float("nan")
                if not mas_points
                else _ext_mod.min_distance_mm(centroid, np.vstack(mas_points), affine)
            )

        vertical_distance = _vert_mod.signed_vertical_distance_mm(
            affine, centroid, left, right, chin, reference_ijk=centroid
        )
        _, foot_to_left_mm, foot_to_right_mm = _lr_mod.project_to_lr_line(
            centroid_w, left_w, right_w
        )
        mid_offset = _lr_mod.foot_lr_mid_offset_ratio(
            lr_line_distance_mm, foot_to_left_mm, foot_to_right_mm
        )

        rows.append(
            {
                "case_id": case_id,
                "stone_id": sid,
                "stone_count_in_image": num_stones,
                **geom,
                "relation": relation,
                "extraction_distance_mm": extraction_distance,
                "min_surface_dist_voxels": closest_min,
                "max_surface_dist_voxels": closest_max,
                "vertical_distance_mm": round(vertical_distance, 3),
                "lr_line_distance_mm": round(lr_line_distance_mm, 3),
                "foot_to_left_mm": round(foot_to_left_mm, 3),
                "foot_to_right_mm": round(foot_to_right_mm, 3),
                "foot_lr_mid_offset_ratio": round(mid_offset, 4),
                "centroid_x": round(float(centroid[0]), 2),
                "centroid_y": round(float(centroid[1]), 2),
                "centroid_z": round(float(centroid[2]), 2),
            }
        )

    return rows


def main() -> None:
    """命令行入口：清洗标签并导出综合特征 CSV。"""
    parser = argparse.ArgumentParser(description="结石综合特征提取（清洗 + 四类特征合并）")
    parser.add_argument(
        "--source-labels-dir",
        type=Path,
        default=DEFAULT_LABELS_DIR,
        help="原始 labelsTr 目录",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="特征提取使用的标签目录；默认使用清洗后的 labelsTr_cleaned",
    )
    parser.add_argument(
        "--cleaned-dir",
        type=Path,
        default=DEFAULT_CLEANED_DIR,
        help="腺体清洗输出目录",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="综合特征 CSV 输出路径",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="跳过清洗步骤，直接使用 --labels-dir 指定目录",
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=MIN_GLAND_VOXELS,
        help="腺体连通域保留阈值（体素数 > 该值）",
    )
    args = parser.parse_args()

    if args.skip_clean:
        labels_dir = args.labels_dir or args.source_labels_dir
        print(f"跳过清洗，使用标签目录: {labels_dir}\n")
    else:
        run_gland_cleaning(args.source_labels_dir, args.cleaned_dir, min_voxels=args.min_voxels)
        labels_dir = args.labels_dir or args.cleaned_dir

    if not labels_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {labels_dir}")

    landmark_order = load_landmark_order(LANDMARK_FILE)
    all_rows: list[dict] = []
    skipped_no_label: list[str] = []
    errors: list[str] = []

    print("提取综合特征...")
    print("=" * 70)
    for case_id in landmark_order:
        label_path = labels_dir / f"image_{case_id}.nii.gz"
        if not label_path.exists():
            skipped_no_label.append(case_id)
            continue
        try:
            rows = analyze_case(label_path)
            all_rows.extend(rows)
            print(f"{case_id}: {len(rows)} stone(s)")
        except Exception as exc:
            errors.append(f"{case_id}: {exc}")
            print(f"{case_id}: ERROR {exc}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print("=" * 70)
    print(f"综合特征行数: {len(all_rows)}")
    print(f"已保存: {args.output_csv}")
    if skipped_no_label:
        print(f"跳过（无标签文件）: {len(skipped_no_label)} 个病例")
    if errors:
        print(f"失败: {len(errors)} 个病例")


if __name__ == "__main__":
    main()
