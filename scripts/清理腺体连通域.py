"""
清理 labelsTr 中腺体（label=2）的小连通域。

规则：
  - 对每个文件中 label=2 做 26-邻接连通域分析
  - 体素数 > 10000 的连通域保留为 label=2
  - 体素数 <= 10000 的连通域改为 label=0（背景）
  - 其他 label（结石、颅骨等）不变

用法：
  python 清理腺体连通域.py
  python 清理腺体连通域.py --dry-run
  python 清理腺体连通域.py --output-dir D:/labelsTr_cleaned
  python 清理腺体连通域.py --backup
"""
import argparse
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

DEFAULT_LABELS_DIR = (
    r"C:\Users\Junbo\PycharmProjects\nnUNet\nnUNet_raw\Dataset001_XXY\labelsTr"
)
GLAND_LABEL = 2
MIN_VOXELS = 10000


def clean_gland_labels(data: np.ndarray, min_voxels: int = MIN_VOXELS) -> tuple[np.ndarray, dict]:
    """
    过滤腺体小连通域。

    Returns:
        新标注数组, 统计信息
    """
    out = data.copy()
    gland_mask = out == GLAND_LABEL
    if not gland_mask.any():
        return out, {
            "component_count": 0,
            "kept_count": 0,
            "removed_count": 0,
            "removed_voxels": 0,
            "kept_voxels": 0,
        }

    labeled, component_count = ndimage.label(gland_mask)
    kept_count = 0
    removed_count = 0
    removed_voxels = 0
    kept_voxels = 0

    for component_id in range(1, component_count + 1):
        component_mask = labeled == component_id
        voxel_count = int(component_mask.sum())
        if voxel_count > min_voxels:
            kept_count += 1
            kept_voxels += voxel_count
        else:
            out[component_mask] = 0
            removed_count += 1
            removed_voxels += voxel_count

    return out, {
        "component_count": component_count,
        "kept_count": kept_count,
        "removed_count": removed_count,
        "removed_voxels": removed_voxels,
        "kept_voxels": kept_voxels,
    }


def process_file(
    label_path: Path,
    output_path: Path,
    min_voxels: int,
    dry_run: bool,
) -> dict:
    img = nib.load(str(label_path))
    data = img.get_fdata().astype(np.int16)
    cleaned, stats = clean_gland_labels(data, min_voxels=min_voxels)

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(cleaned, img.affine, img.header), str(output_path))

    stats["file"] = label_path.name
    stats["output"] = str(output_path)
    return stats


def backup_directory(labels_dir: Path) -> Path:
    backup_dir = labels_dir.parent / f"{labels_dir.name}_backup"
    if backup_dir.exists():
        raise FileExistsError(
            f"备份目录已存在，请先删除或改名: {backup_dir}"
        )
    shutil.copytree(labels_dir, backup_dir)
    return backup_dir


def main():
    parser = argparse.ArgumentParser(description="清理腺体 label 的小连通域")
    parser.add_argument(
        "labels_dir",
        nargs="?",
        default=DEFAULT_LABELS_DIR,
        help="labelsTr 目录",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录；默认覆盖原 labelsTr",
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=MIN_VOXELS,
        help=f"保留连通域的最小体素数阈值，默认 {MIN_VOXELS}（保留 > 该值）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计，不写文件",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="处理前备份整个 labelsTr 到 labelsTr_backup",
    )
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        print(f"错误：目录不存在 - {labels_dir}")
        sys.exit(1)

    label_files = sorted(labels_dir.glob("*.nii.gz"))
    if not label_files:
        print(f"错误：未找到 .nii.gz 文件 - {labels_dir}")
        sys.exit(1)

    in_place = args.output_dir is None
    output_dir = labels_dir if in_place else Path(args.output_dir)

    if args.backup and not args.dry_run:
        backup_path = backup_directory(labels_dir)
        print(f"已备份到: {backup_path}\n")

    print(f"输入目录: {labels_dir}")
    print(f"输出目录: {output_dir}")
    print(f"阈值: 保留体素数 > {args.min_voxels} 的腺体连通域")
    if args.dry_run:
        print("模式: dry-run（不写入文件）")
    print("=" * 70)

    total_removed_components = 0
    total_removed_voxels = 0

    for label_path in label_files:
        output_path = output_dir / label_path.name
        stats = process_file(
            label_path,
            output_path,
            min_voxels=args.min_voxels,
            dry_run=args.dry_run,
        )
        total_removed_components += stats["removed_count"]
        total_removed_voxels += stats["removed_voxels"]
        print(
            f"{stats['file']}: "
            f"连通域 {stats['component_count']} -> 保留 {stats['kept_count']}, "
            f"删除 {stats['removed_count']} 个, "
            f"清除 {stats['removed_voxels']} 体素, "
            f"保留 {stats['kept_voxels']} 体素"
        )

    print("=" * 70)
    print(f"处理文件数: {len(label_files)}")
    print(f"共删除小连通域: {total_removed_components} 个")
    print(f"共清除体素: {total_removed_voxels}")
    if args.dry_run:
        print("未写入任何文件（dry-run）")
    elif in_place:
        print("已原地更新 labelsTr")
    else:
        print(f"已保存到: {output_dir}")


if __name__ == "__main__":
    main()
