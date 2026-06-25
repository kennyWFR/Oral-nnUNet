"""
convert_to_lps.py
检查 imagesTr 目录下所有图像是否为 LPS 朝向，不是则转换
"""
import os
import glob
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation


def get_orientation(img):
    """获取图像的朝向（orientation codes）"""
    aff = img.affine
    return nib.aff2axcodes(aff)


def convert_to_lps(img_path, save_path=None):
    """
    将图像转换为 LPS 朝向
    LPS = Left, Posterior, Superior
    """
    if save_path is None:
        save_path = img_path

    img = nib.load(img_path)
    current_orient = get_orientation(img)
    target_orient = ('L', 'P', 'S')

    print(f"文件: {os.path.basename(img_path)}")
    print(f"  当前朝向: {current_orient}")

    if current_orient == target_orient:
        print(f"  ✓ 已是 LPS，无需转换")
        return False

    print(f"  目标朝向: {target_orient}")
    print(f"  → 正在转换...")

    # 计算从当前朝向到 LPS 的转换
    current_ornt = axcodes2ornt(current_orient)
    target_ornt = axcodes2ornt(target_orient)
    transform = ornt_transform(current_ornt, target_ornt)

    # 应用转换
    img_reoriented = apply_orientation(img, transform)

    # 保存
    nib.save(img_reoriented, save_path)
    print(f"  ✓ 已保存为 LPS")
    return True


def process_directory(directory):
    """处理目录下所有 .nii.gz 文件"""
    pattern = os.path.join(directory, "*.nii.gz")
    files = glob.glob(pattern)

    if not files:
        print(f"未找到 .nii.gz 文件: {directory}")
        return

    print(f"找到 {len(files)} 个文件\n")

    converted_count = 0
    for file_path in sorted(files):
        try:
            if convert_to_lps(file_path):
                converted_count += 1
        except Exception as e:
            print(f"  ✗ 错误: {e}")
        print()

    print("=" * 50)
    print(f"处理完成: {len(files)} 个文件")
    print(f"已转换: {converted_count} 个")
    print(f"无需转换: {len(files) - converted_count} 个")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = r"C:\Users\Junbo\PycharmProjects\nnUNet\nnUNet_raw\Dataset001_XXY\imagesTr"

    process_directory(directory)