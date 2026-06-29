"""
对单个 NIfTI CT 图像运行 Dataset001_XXY nnUNet 推理。
供 3D Slicer 模块、ITK-SNAP 工作流等外部调用。

用法:
  python itksnap_nnunet_predict.py -i input.nii.gz
  python itksnap_nnunet_predict.py -i input.nii.gz -o output_seg.nii.gz --device cuda
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_FOLDER = (
    PROJECT_ROOT
    / "nnUNet_results"
    / "Dataset001_XXY"
    / "nnUNetTrainer__nnUNetPlans__3d_fullres"
)
CHECKPOINT_NAME = "checkpoint_final.pth"
USE_FOLDS = (0,)


def setup_nnunet_env(project_root: Path) -> None:
    os.environ["nnUNet_results"] = str(project_root / "nnUNet_results")
    os.environ["nnUNet_raw"] = str(project_root / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(project_root / "nnUNet_preprocessed")


def stem_without_nii(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def default_output_path(input_image: Path) -> Path:
    stem = stem_without_nii(input_image)
    if stem.endswith("_0000"):
        stem = stem[:-5]
    return input_image.with_name(f"{stem}_nnunet_seg.nii.gz")


def build_predictor(model_folder: Path, device_name: str) -> nnUNetPredictor:
    checkpoint = model_folder / "fold_0" / CHECKPOINT_NAME
    if not checkpoint.is_file():
        raise FileNotFoundError(f"未找到模型权重: {checkpoint}")

    if device_name == "cuda" and not torch.cuda.is_available():
        print("警告：CUDA 不可用，自动切换到 CPU（会很慢）")
        device_name = "cpu"

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=device_name == "cuda",
        device=torch.device(device_name),
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=USE_FOLDS,
        checkpoint_name=CHECKPOINT_NAME,
    )
    return predictor


def run_inference(
    input_image: Path,
    output_seg: Path,
    model_folder: Path,
    device_name: str,
) -> None:
    input_image = input_image.resolve()
    output_seg = output_seg.resolve()
    output_seg.parent.mkdir(parents=True, exist_ok=True)

    case_id = stem_without_nii(input_image)
    if case_id.endswith("_0000"):
        case_id = case_id[:-5]

    with tempfile.TemporaryDirectory(prefix="nnunet_predict_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        tmp_input = tmp_dir_path / f"{case_id}_0000.nii.gz"
        shutil.copy2(input_image, tmp_input)

        output_truncated = output_seg
        if output_truncated.name.endswith(".nii.gz"):
            output_truncated = output_truncated.with_name(output_truncated.name[:-7])
        elif output_truncated.name.endswith(".nii"):
            output_truncated = output_truncated.with_name(output_truncated.name[:-4])

        predictor = build_predictor(model_folder, device_name)
        predictor.predict_from_files(
            [[str(tmp_input)]],
            [str(output_truncated)],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )

    if not output_seg.is_file():
        raise FileNotFoundError(f"推理完成但未找到输出文件: {output_seg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset001_XXY 单图 nnUNet 推理")
    parser.add_argument("-i", "--input", required=True, help="输入 CT NIfTI 路径")
    parser.add_argument("-o", "--output", default=None, help="输出分割路径")
    parser.add_argument(
        "--model-folder",
        default=str(DEFAULT_MODEL_FOLDER),
        help="nnUNet 模型目录",
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="nnUNet 项目根目录（用于环境变量）",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="推理设备",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_image = Path(args.input)
    if not input_image.is_file():
        print(f"错误：输入文件不存在 - {input_image}", file=sys.stderr)
        sys.exit(1)

    model_folder = Path(args.model_folder)
    if not model_folder.is_dir():
        print(f"错误：模型目录不存在 - {model_folder}", file=sys.stderr)
        sys.exit(1)

    output_seg = Path(args.output) if args.output else default_output_path(input_image)
    setup_nnunet_env(Path(args.project_root))

    print(f"输入: {input_image}")
    print(f"输出: {output_seg}")
    print(f"模型: {model_folder}")
    print("开始推理...")

    run_inference(input_image, output_seg, model_folder, args.device)
    print(f"分割结果已保存: {output_seg}")


if __name__ == "__main__":
    main()
