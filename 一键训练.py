"""
一键启动 nnUNet 训练（Dataset001_XXY）

默认流程：
1. 设置 nnUNet_raw / nnUNet_preprocessed / nnUNet_results
2. 若尚未预处理，自动运行 plan_and_preprocess
3. 启动 3d_fullres 训练（fold 0）

在 PyCharm 中选择 nnunet 解释器后直接运行本脚本即可。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ============ 可按需修改 ============
DATASET_ID = 1
DATASET_NAME = "Dataset001_XXY"
CONFIGURATION = "3d_fullres"   # 可选: 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres
FOLD = 0                       # 0~4；设为 "all" 则用全部训练样本训练单模型
USE_NPZ = True                 # 建议开启，便于后续选最佳配置和集成
DEVICE = "cuda"                # cuda / cpu / mps
RUN_PREPROCESS_IF_NEEDED = True
VERIFY_DATASET_ON_PREPROCESS = True
# ===================================

PROJECT_ROOT = Path(__file__).resolve().parent
NNUNET_RAW = PROJECT_ROOT / "nnUNet_raw"
NNUNET_PREPROCESSED = PROJECT_ROOT / "nnUNet_preprocessed"
NNUNET_RESULTS = PROJECT_ROOT / "nnUNet_results"

CLI_FALLBACK = {
    "nnUNetv2_train": PROJECT_ROOT / "nnunetv2" / "run" / "run_training.py",
    "nnUNetv2_plan_and_preprocess": PROJECT_ROOT
    / "nnunetv2"
    / "experiment_planning"
    / "plan_and_preprocess_entrypoints.py",
}


def setup_environment() -> None:
    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)
    NNUNET_RESULTS.mkdir(parents=True, exist_ok=True)


def build_nnunet_command(name: str, args: list[str]) -> list[str]:
    candidate = Path(sys.executable).parent / f"{name}.exe"
    if candidate.exists():
        return [str(candidate), *args]

    found = shutil.which(name)
    if found:
        return [found, *args]

    fallback = CLI_FALLBACK.get(name)
    if fallback and fallback.exists():
        return [sys.executable, str(fallback), *args]

    raise FileNotFoundError(
        f"找不到命令 {name}。请确认已用 nnunet 环境执行: pip install -e ."
    )


def run_command(cmd: list[str], title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
    print("命令:", " ".join(cmd))
    print()
    subprocess.run(cmd, check=True)


def is_preprocessed() -> bool:
    plans_file = NNUNET_PREPROCESSED / DATASET_NAME / "nnUNetPlans.json"
    return plans_file.exists()


def run_preprocess() -> None:
    args = ["-d", str(DATASET_ID)]
    if VERIFY_DATASET_ON_PREPROCESS:
        args.append("--verify_dataset_integrity")
    cmd = build_nnunet_command("nnUNetv2_plan_and_preprocess", args)
    run_command(cmd, "步骤 1/2: 规划与预处理")


def build_train_command() -> list[str]:
    args = [str(DATASET_ID), CONFIGURATION, str(FOLD), "-device", DEVICE]
    if USE_NPZ:
        args.append("--npz")
    return build_nnunet_command("nnUNetv2_train", args)


def print_summary() -> None:
    print()
    print("=" * 60)
    print("环境配置")
    print("=" * 60)
    print(f"Python:           {sys.executable}")
    print(f"nnUNet_raw:       {os.environ['nnUNet_raw']}")
    print(f"nnUNet_preprocessed:{os.environ['nnUNet_preprocessed']}")
    print(f"nnUNet_results:   {os.environ['nnUNet_results']}")
    print(f"数据集:           {DATASET_NAME} (ID={DATASET_ID})")
    print(f"配置:             {CONFIGURATION}")
    print(f"Fold:             {FOLD}")
    print(f"设备:             {DEVICE}")
    print()
    print("训练结果将保存到:")
    print(
        NNUNET_RESULTS
        / DATASET_NAME
        / f"nnUNetTrainer__nnUNetPlans__{CONFIGURATION}"
        / f"fold_{FOLD}"
    )
    print("=" * 60)
    print()


def main() -> None:
    setup_environment()
    print_summary()

    if RUN_PREPROCESS_IF_NEEDED and not is_preprocessed():
        run_preprocess()
    elif is_preprocessed():
        print("检测到已完成预处理，跳过 plan_and_preprocess。\n")
    else:
        print("警告: 未检测到预处理结果，且 RUN_PREPROCESS_IF_NEEDED=False。\n")

    train_cmd = build_train_command()
    run_command(train_cmd, "步骤 2/2: 开始训练")

    print()
    print("训练已启动并完成当前 fold。")
    print("查看训练曲线: 上述 fold 目录中的 progress.png")
    print()
    print("继续训练其他 fold 示例:")
    print(f'  修改脚本中 FOLD = 1，或命令行: nnUNetv2_train {DATASET_ID} {CONFIGURATION} 1 --npz')


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\n命令执行失败，退出码: {exc.returncode}")
        sys.exit(exc.returncode)
    except FileNotFoundError as exc:
        print(f"\n错误: {exc}")
        sys.exit(1)
