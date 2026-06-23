"""
修复 labelsTr 中的标签文件：
1. 从 git 暂存区恢复被清空的 mask 文件
2. 将未压缩的 .nii 内容正确保存为 .nii.gz
3. 统一重命名为 image_{数字}.nii.gz
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib

REPO_ROOT = Path(r"C:\Users\Junbo\PycharmProjects\nnUNet")
LABELS_DIR = REPO_ROOT / r"nnUNet_raw\Dataset001_XXY\labelsTr"
GIT_LABELS_PREFIX = "nnUNet_raw/Dataset001_XXY/labelsTr/"


def _target_label_name(number):
    return f"image_{number}.nii.gz"


def _extract_label_number(base_name):
    patterns = [
        r"^image_(\d+)$",
        r"^image(\d+)$",
        r"^mask_(\d+)$",
        r"^mask(\d+)$",
        r"^label_(\d+)$",
        r"^label(\d+)$",
    ]
    for pattern in patterns:
        match = re.fullmatch(pattern, base_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _save_as_nifti_gz(data: bytes, output_path: Path):
    if len(data) == 0:
        raise ValueError("empty file")

    if data[:2] == b"\x1f\x8b":
        output_path.write_bytes(data)
        return

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        img = nib.load(tmp_path)
        nib.save(img, str(output_path))
    finally:
        os.remove(tmp_path)


def _git_staged_label_files():
    result = subprocess.run(
        ["git", "ls-files", GIT_LABELS_PREFIX],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _read_git_blob(git_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f":{git_path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))
    return result.stdout


def restore_from_git():
    restored = 0
    for git_path in _git_staged_label_files():
        filename = Path(git_path).name
        base_name = filename.replace(".nii.gz", "").replace(".nii", "")
        number = _extract_label_number(base_name)
        if number is None:
            print(f"跳过 git 文件（无法识别编号）: {filename}")
            continue

        target_name = _target_label_name(number)
        target_path = LABELS_DIR / target_name
        data = _read_git_blob(git_path)
        _save_as_nifti_gz(data, target_path)
        print(f"从 git 恢复: {filename} -> {target_name} ({len(data)} bytes)")
        restored += 1
    return restored


def fix_existing_labels():
    fixed = 0
    for path in sorted(LABELS_DIR.glob("image_*.nii*")):
        data = path.read_bytes()
        if len(data) == 0:
            continue
        if data[:2] == b"\x1f\x8b":
            continue

        print(f"修复格式: {path.name}（未压缩 .nii 误命名为 .nii.gz）")
        _save_as_nifti_gz(data, path)
        fixed += 1
    return fixed


def verify_labels():
    ok = 0
    bad = []
    for path in sorted(LABELS_DIR.glob("image_*.nii.gz")):
        data = path.read_bytes()
        if len(data) == 0:
            bad.append(f"{path.name}: 空文件")
            continue
        try:
            nib.load(str(path))
            ok += 1
        except Exception as exc:
            bad.append(f"{path.name}: {exc}")
    return ok, bad


def main():
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("步骤 1: 从 git 暂存区恢复 mask 文件")
    print("=" * 60)
    restored = restore_from_git()

    print("\n" + "=" * 60)
    print("步骤 2: 修复未正确压缩的 label 文件")
    print("=" * 60)
    fixed = fix_existing_labels()

    print("\n" + "=" * 60)
    print("步骤 3: 验证所有 label 文件")
    print("=" * 60)
    ok, bad = verify_labels()
    print(f"可读文件: {ok}")
    print(f"问题文件: {len(bad)}")
    for item in bad:
        print(f"  - {item}")

    print("\n" + "=" * 60)
    print(f"完成: 从 git 恢复 {restored} 个, 修复格式 {fixed} 个")
    print("=" * 60)


if __name__ == "__main__":
    main()
