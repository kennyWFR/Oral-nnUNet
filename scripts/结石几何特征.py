"""
提取 labelsTr 中 label=1（结石）的几何特征，并导出 CSV。
每个连通域（单颗结石）占一行；若病例无结石，输出一行空特征记录。

结石几何特征指标说明（CSV 列名对照表）
========================================================================================================================
| CSV 列名                  | 中文名称         | 含义                                      | 临床/研究意义                                      |
|---------------------------|------------------|-------------------------------------------|----------------------------------------------------|
| image_id                  | 病例标识         | 标签文件名对应的病例名                    | 关联影像、临床资料与分割结果                       |
| stone_id                  | 结石编号         | 同一病例内第几颗结石（连通域编号）        | 区分单发/多发结石中的每一颗                        |
| stone_count_in_image      | 结石个数         | 该病例中 label=1 的连通域总数             | 评估结石负荷、多发结石情况                         |
| volume_voxels             | 体素数           | 结石占据的体素数量                        | 原始计数，用于核对与质量控制                       |
| volume_mm3                | 体积             | 结石物理体积（mm³）                       | 最基础量化指标，反映结石大小与总负荷               |
| surface_area_mm2          | 表面积           | 结石三维表面面积（mm²）                   | 与药物溶解、体外碎石接触面积等相关                 |
| equivalent_diameter_mm    | 等效直径         | 与结石同体积球体的直径（mm）              | 常用汇总“结石大小”的单一数值                       |
| max_feret_diameter_mm     | 最大径           | 结石表面任意两点间的最大距离（mm）        | 临床最常报告的“结石最大径”，与治疗决策密切相关     |
| major_axis_length_mm      | 长轴长度         | 基于惯性张量的主轴长度（mm）              | 描述结石主要延伸方向与最长尺度                     |
| minor_axis_length_mm      | 短轴长度         | 基于惯性张量的次轴长度（mm）              | 描述结石较窄方向的尺度                             |
| axis_ratio_major_minor    | 长短轴比         | 长轴长度 ÷ 短轴长度                       | 越大表示结石越细长，反映形态不规则程度             |
| bbox_length_x_mm          | 包围盒 X 边长    | 结石外接框在 X 方向的物理边长（mm）       | 粗略描述结石在体素 X 方向的空间占位                 |
| bbox_length_y_mm          | 包围盒 Y 边长    | 结石外接框在 Y 方向的物理边长（mm）       | 粗略描述结石在体素 Y 方向的空间占位                 |
| bbox_length_z_mm          | 包围盒 Z 边长    | 结石外接框在 Z 方向的物理边长（mm）       | 粗略描述结石在体素 Z 方向的空间占位                 |
| bbox_diagonal_mm          | 包围盒对角线     | 三维外接框对角线长度（mm）                | 整体空间尺度的参考指标                             |
| sphericity                | 球形度           | 接近理想球形的程度，1 表示完美球体        | 形态越规则越接近 1，反映结石形状规则性             |
| compactness               | 紧密度           | 体积与表面积的综合比值                    | 值越大通常表示形状越紧凑                           |
| centroid_x_mm             | 质心 X 坐标      | 结石质心在物理空间 X 方向的位置（mm）     | 结石空间定位，可用于左右侧/解剖位置分析             |
| centroid_y_mm             | 质心 Y 坐标      | 结石质心在物理空间 Y 方向的位置（mm）     | 结石空间定位，可用于前后/上下位置分析               |
| centroid_z_mm             | 质心 Z 坐标      | 结石质心在物理空间 Z 方向的位置（mm）     | 结石空间定位，可用于层面分布分析                     |
========================================================================================================================

公式参考：
  - equivalent_diameter_mm = (6 × volume_mm3 / π)^(1/3)
  - sphericity             = π^(1/3) × (6 × volume_mm3)^(2/3) / surface_area_mm2
  - compactness            = volume_mm3 / (surface_area_mm2 ^ 1.5)
  - max_feret_diameter_mm  = marching cubes 网格顶点间最大欧氏距离
"""
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import measure


DEFAULT_LABELS_DIR = (
    r"C:\Users\Junbo\PycharmProjects\nnUNet\nnUNet_raw\Dataset001_XXY\labelsTr"
)
DEFAULT_OUTPUT_CSV = (
    r"C:\Users\Junbo\PycharmProjects\nnUNet\nnUNet_raw\Dataset001_XXY\stone_geometry_features.csv"
)
STONE_LABEL = 1

# 输出 CSV 列定义，与上方模块文档中的指标对照表一一对应
CSV_COLUMNS = [
    "image_id",
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
]


def image_id_from_path(path: Path) -> str:
    """从标签文件路径提取病例标识（去掉 .nii.gz / .nii 扩展名）。"""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def empty_feature_row(image_id: str) -> dict:
    """生成无结石病例的空特征行（stone_id=0，其余字段留空）。"""
    row = {col: "" for col in CSV_COLUMNS}
    row["image_id"] = image_id
    row["stone_id"] = 0
    row["stone_count_in_image"] = 0
    return row


def compute_stone_features(stone_mask: np.ndarray, spacing: tuple[float, float, float]) -> dict:
    """对单颗结石的二值 mask 计算体积、表面积、形态与质心等几何特征。"""
    volume_voxels = int(stone_mask.sum())
    voxel_volume = float(np.prod(spacing))
    volume_mm3 = volume_voxels * voxel_volume

    equivalent_diameter_mm = (6.0 * volume_mm3 / np.pi) ** (1.0 / 3.0)

    surface_area_mm2 = np.nan
    max_feret_diameter_mm = np.nan
    try:
        verts, faces, _, _ = measure.marching_cubes(
            stone_mask.astype(np.float32),
            level=0.5,
            spacing=spacing,
        )
        surface_area_mm2 = float(measure.mesh_surface_area(verts, faces))
        if len(verts) >= 2:
            from scipy.spatial.distance import pdist

            max_feret_diameter_mm = float(pdist(verts).max())
    except (ValueError, RuntimeError):
        pass

    labeled = stone_mask.astype(np.uint8)
    props = measure.regionprops(labeled, spacing=spacing)[0]

    major_axis = float(props.axis_major_length)
    minor_axis = float(props.axis_minor_length)
    axis_ratio = major_axis / minor_axis if minor_axis > 0 else np.nan

    min_row, min_col, min_slice, max_row, max_col, max_slice = props.bbox
    bbox_length_x = (max_row - min_row) * spacing[0]
    bbox_length_y = (max_col - min_col) * spacing[1]
    bbox_length_z = (max_slice - min_slice) * spacing[2]
    bbox_diagonal = float(
        np.sqrt(bbox_length_x ** 2 + bbox_length_y ** 2 + bbox_length_z ** 2)
    )

    if surface_area_mm2 > 0 and volume_mm3 > 0:
        sphericity = float(
            (np.pi ** (1.0 / 3.0) * (6.0 * volume_mm3) ** (2.0 / 3.0)) / surface_area_mm2
        )
        compactness = float(volume_mm3 / (surface_area_mm2 ** 1.5))
    else:
        sphericity = np.nan
        compactness = np.nan

    centroid = props.centroid

    return {
        "volume_voxels": volume_voxels,
        "volume_mm3": round(volume_mm3, 4),
        "surface_area_mm2": round(surface_area_mm2, 4) if np.isfinite(surface_area_mm2) else "",
        "equivalent_diameter_mm": round(equivalent_diameter_mm, 4),
        "max_feret_diameter_mm": round(max_feret_diameter_mm, 4)
        if np.isfinite(max_feret_diameter_mm)
        else "",
        "major_axis_length_mm": round(major_axis, 4),
        "minor_axis_length_mm": round(minor_axis, 4),
        "axis_ratio_major_minor": round(axis_ratio, 4) if np.isfinite(axis_ratio) else "",
        "bbox_length_x_mm": round(bbox_length_x, 4),
        "bbox_length_y_mm": round(bbox_length_y, 4),
        "bbox_length_z_mm": round(bbox_length_z, 4),
        "bbox_diagonal_mm": round(bbox_diagonal, 4),
        "sphericity": round(sphericity, 4) if np.isfinite(sphericity) else "",
        "compactness": round(compactness, 6) if np.isfinite(compactness) else "",
        "centroid_x_mm": round(float(centroid[0]), 4),
        "centroid_y_mm": round(float(centroid[1]), 4),
        "centroid_z_mm": round(float(centroid[2]), 4),
    }


def extract_features_from_file(label_path: Path) -> list[dict]:
    """读取单个标签文件，提取其中所有结石连通域的特征行；无结石时返回一行空记录。"""
    image_id = image_id_from_path(label_path)
    img = nib.load(str(label_path))
    data = img.get_fdata()
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])

    stone_mask = data == STONE_LABEL
    if not stone_mask.any():
        return [empty_feature_row(image_id)]

    labeled, stone_count = ndimage.label(stone_mask)
    rows = []

    for stone_id in range(1, stone_count + 1):
        component_mask = labeled == stone_id
        features = compute_stone_features(component_mask, spacing)
        row = {
            "image_id": image_id,
            "stone_id": stone_id,
            "stone_count_in_image": stone_count,
            **features,
        }
        rows.append(row)

    return rows


def process_directory(labels_dir: Path, output_csv: Path) -> None:
    """批量处理目录下全部标签文件，汇总特征并写入 CSV。"""
    label_files = sorted(labels_dir.glob("*.nii.gz"))
    if not label_files:
        raise FileNotFoundError(f"未找到 .nii.gz 文件: {labels_dir}")

    all_rows = []
    for label_path in label_files:
        print(f"处理: {label_path.name}")
        all_rows.extend(extract_features_from_file(label_path))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    images_with_stone = len({r["image_id"] for r in all_rows if r["stone_count_in_image"] != 0})
    total_stones = sum(
        1 for r in all_rows if r["stone_count_in_image"] != 0 and r["stone_id"] != 0
    )
    print("=" * 60)
    print(f"标签目录: {labels_dir}")
    print(f"处理病例数: {len(label_files)}")
    print(f"含结石病例数: {images_with_stone}")
    print(f"结石总数: {total_stones}")
    print(f"CSV 已保存: {output_csv}")


def main():
    """命令行入口：解析 labels 目录与输出路径，调用 process_directory。"""
    labels_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_LABELS_DIR)
    output_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(DEFAULT_OUTPUT_CSV)
    process_directory(labels_dir, output_csv)


if __name__ == "__main__":
    main()
