"""
analyze_labels.py
统计 NIfTI 分割文件中所有 label 种类和数量
"""
import sys
import numpy as np
import nibabel as nib


def analyze_labels(nii_path):
    """读取 mask 文件，统计每个 label 的种类和数量"""
    img = nib.load(nii_path)
    data = img.get_fdata().astype(np.int32)

    # 找出所有唯一值和对应的数量
    labels, counts = np.unique(data, return_counts=True)

    print(f"文件: {nii_path}")
    print(f"图像尺寸: {data.shape}")
    print(f"像素总数: {data.size}")
    print("-" * 50)
    print(f"{'Label 值':<12} {'像素数量':<15} {'占比'}")
    print("-" * 50)

    for label, count in zip(labels, counts):
        percentage = count / data.size * 100
        print(f"{label:<12} {count:<15} {percentage:.2f}%")

    print("-" * 50)
    print(f"Label 种类总数: {len(labels)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = r"C:\Users\Junbo\PycharmProjects\nnUNet\nnUNet_raw\Dataset001_XXY\labelsTr\mask3.nii.gz"

    analyze_labels(path)