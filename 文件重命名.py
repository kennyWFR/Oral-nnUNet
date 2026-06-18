"""
重命名 Dataset001_XXY 中的图像和标签文件，使其符合 nnU-Net 命名规范。

imagesTr:
    image3.nii.gz / image_3.nii.gz / image7.nii -> image_3_0000.nii.gz

labelsTr:
    mask3.nii.gz / image_3.nii.gz / image3301546196.nii -> image_3.nii.gz

.nii 转 .nii.gz 时会用 nibabel 真正压缩，不会只改扩展名。
"""
import os
import re
import sys
import tempfile

import numpy as np
import nibabel as nib

DEFAULT_DATASET_DIR = r"C:\Users\Junbo\PycharmProjects\nnUNet\nnUNet_raw\Dataset001_XXY"
NIFTI_EXTENSIONS = (".nii.gz", ".nii")


def _split_nifti_filename(filename):
    for ext in NIFTI_EXTENSIONS:
        if filename.endswith(ext):
            return filename[: -len(ext)], ext
    return None, None


def _target_image_name(number):
    return f"image_{number}_0000.nii.gz"


def _target_label_name(number):
    return f"image_{number}.nii.gz"


def _is_standard_image_name(base_name):
    return re.fullmatch(r"image_\d+_0000", base_name, re.IGNORECASE) is not None


def _is_standard_label_name(base_name):
    return re.fullmatch(r"image_\d+", base_name, re.IGNORECASE) is not None


def _extract_image_number(base_name):
    patterns = [
        r"^image_(\d+)$",
        r"^image(\d+)$",
    ]
    for pattern in patterns:
        match = re.fullmatch(pattern, base_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _extract_label_number(base_name):
    patterns = [
        r"^image_(\d+)$",
        r"^image(\d+)$",
        r"^label_(\d+)$",
        r"^label(\d+)$",
        r"^mask_(\d+)$",
        r"^mask(\d+)$",
    ]
    for pattern in patterns:
        match = re.fullmatch(pattern, base_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _list_nifti_files(directory):
    if not os.path.exists(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.endswith(NIFTI_EXTENSIONS))


def count_images_tr(directory):
    """统计 imagesTr 目录下的文件数量并打印"""
    print("[imagesTr] 文件统计")
    print("=" * 60)

    if not os.path.exists(directory):
        print(f"错误：目录不存在 - {directory}\n")
        return 0

    files = _list_nifti_files(directory)
    nii_gz_count = sum(1 for f in files if f.endswith(".nii.gz"))
    nii_count = sum(1 for f in files if f.endswith(".nii") and not f.endswith(".nii.gz"))
    standard_count = sum(
        1 for f in files if _is_standard_image_name(_split_nifti_filename(f)[0])
    )

    print(f"目录: {directory}")
    print(f"文件总数: {len(files)}")
    print(f"  .nii.gz: {nii_gz_count}")
    print(f"  .nii:    {nii_count}")
    print(f"  标准格式 (image_X_0000): {standard_count}")
    print(f"  非标准格式: {len(files) - standard_count}")
    print("=" * 60 + "\n")
    return len(files)


def _is_real_gzip(path):
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def _eager_load(img, as_label=False):
    """将懒加载图像读入内存，避免临时文件删除后无法保存。"""
    data = img.get_fdata()
    if as_label:
        data = np.rint(data).astype(np.uint8)
        header = img.header.copy()
        header.set_data_dtype(np.uint8)
        return nib.Nifti1Image(data, img.affine, header)
    return nib.Nifti1Image(data, img.affine, img.header)


def _load_nifti(path, as_label=False):
    if _is_real_gzip(path):
        return _eager_load(nib.load(path), as_label=as_label)

    _, ext = _split_nifti_filename(os.path.basename(path))
    if ext == ".nii":
        return _eager_load(nib.load(path), as_label=as_label)

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        tmp_path = tmp.name
        with open(path, "rb") as src:
            tmp.write(src.read())

    try:
        return _eager_load(nib.load(tmp_path), as_label=as_label)
    finally:
        os.remove(tmp_path)


def _needs_gzip_conversion(filename, path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False

    _, old_ext = _split_nifti_filename(filename)
    if old_ext == ".nii":
        return True
    return not _is_real_gzip(path)


def _rename_sort_key(directory, filename):
    path = os.path.join(directory, filename)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    _, ext = _split_nifti_filename(filename)
    ext_priority = 0 if ext == ".nii" else 1
    return ext_priority, -size


def _write_nifti_gz(src_path, dst_path, as_label=False):
    img = _load_nifti(src_path, as_label=as_label)
    src_abs = os.path.abspath(src_path)
    dst_abs = os.path.abspath(dst_path)

    if src_abs == dst_abs:
        dst_dir = os.path.dirname(dst_path)
        fd, tmp_path = tempfile.mkstemp(suffix=".nii.gz", dir=dst_dir)
        os.close(fd)
        try:
            nib.save(img, tmp_path)
            os.replace(tmp_path, dst_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        return

    nib.save(img, dst_path)
    os.remove(src_path)


def _rename_file(directory, filename, new_name, rename_count, skip_count, as_label=False):
    if filename == new_name:
        old_path = os.path.join(directory, filename)
        if _needs_gzip_conversion(filename, old_path):
            print(f"修复格式: {filename}")
            print("  → 重新保存为真正的 .nii.gz\n")
            _write_nifti_gz(old_path, old_path, as_label=as_label)
            return rename_count + 1, skip_count
        print(f"跳过: {filename}")
        print("  → 已是标准格式\n")
        return rename_count, skip_count + 1

    old_path = os.path.join(directory, filename)
    new_path = os.path.join(directory, new_name)
    old_size = os.path.getsize(old_path) if os.path.exists(old_path) else 0

    if os.path.exists(new_path):
        new_size = os.path.getsize(new_path)
        if new_size > 0:
            print(f"警告: 目标文件已存在 - {new_name}")
            print(f"  → 跳过 {filename}\n")
            return rename_count, skip_count
        if old_size == 0:
            print(f"警告: 源文件为空 - {filename}")
            print("  → 跳过\n")
            return rename_count, skip_count
        print(f"覆盖空文件: {new_name} <- {filename}")
        os.remove(new_path)

    if _needs_gzip_conversion(filename, old_path):
        print(f"转换并重命名: {filename}")
        print(f"  → {new_name}\n")
        _write_nifti_gz(old_path, new_path, as_label=as_label)
    else:
        os.rename(old_path, new_path)
        print(f"重命名: {filename}")
        print(f"  → {new_name}\n")
    return rename_count + 1, skip_count


def rename_images(directory):
    if not os.path.exists(directory):
        print(f"错误：目录不存在 - {directory}")
        return

    files = sorted(_list_nifti_files(directory), key=lambda f: _rename_sort_key(directory, f))
    if not files:
        print(f"未找到 NIfTI 文件: {directory}")
        return

    print(f"[imagesTr] 开始重命名，共 {len(files)} 个文件\n")
    print("=" * 60)

    rename_count = 0
    skip_count = 0

    for filename in files:
        base_name, _ = _split_nifti_filename(filename)

        if _is_standard_image_name(base_name):
            rename_count, skip_count = _rename_file(
                directory, filename, filename, rename_count, skip_count
            )
            continue

        number = _extract_image_number(base_name)
        if number is None:
            print(f"警告: 无法识别的文件名 - {filename}")
            print("  → 请手动处理\n")
            continue

        new_name = _target_image_name(number)
        rename_count, skip_count = _rename_file(
            directory, filename, new_name, rename_count, skip_count
        )

    print("=" * 60)
    print("imagesTr 处理完成:")
    print(f"  已重命名: {rename_count} 个文件")
    print(f"  已跳过: {skip_count} 个文件")
    print(f"  总计: {len(files)} 个文件\n")


def rename_labels(directory):
    if not os.path.exists(directory):
        print(f"错误：目录不存在 - {directory}")
        return

    files = sorted(_list_nifti_files(directory), key=lambda f: _rename_sort_key(directory, f))
    if not files:
        print(f"未找到 NIfTI 文件: {directory}")
        return

    print(f"[labelsTr] 开始重命名，共 {len(files)} 个文件\n")
    print("=" * 60)

    rename_count = 0
    skip_count = 0

    for filename in files:
        base_name, _ = _split_nifti_filename(filename)

        if _is_standard_label_name(base_name):
            new_name = _target_label_name(int(re.search(r"\d+", base_name).group()))
            rename_count, skip_count = _rename_file(
                directory, filename, new_name, rename_count, skip_count, as_label=True
            )
            continue

        number = _extract_label_number(base_name)
        if number is None:
            print(f"警告: 无法识别的文件名 - {filename}")
            print("  → 请手动处理\n")
            continue

        new_name = _target_label_name(number)
        rename_count, skip_count = _rename_file(
            directory, filename, new_name, rename_count, skip_count, as_label=True
        )

    print("=" * 60)
    print("labelsTr 处理完成:")
    print(f"  已重命名: {rename_count} 个文件")
    print(f"  已跳过: {skip_count} 个文件")
    print(f"  总计: {len(files)} 个文件\n")


def main(dataset_dir):
    images_dir = os.path.join(dataset_dir, "imagesTr")
    labels_dir = os.path.join(dataset_dir, "labelsTr")

    print(f"数据集目录: {dataset_dir}\n")
    count_images_tr(images_dir)
    rename_images(images_dir)
    rename_labels(labels_dir)
    print("重命名后统计:")
    count_images_tr(images_dir)


if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET_DIR
    main(dataset_dir)
