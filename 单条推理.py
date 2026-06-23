import os
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

os.environ["nnUNet_results"] = r"C:\Users\Junbo\PycharmProjects\nnUNet\nnUNet_results"
os.environ["nnUNet_raw"] = r"C:\Users\Junbo\PycharmProjects\nnUNet\nnUNet_raw"
os.environ["nnUNet_preprocessed"] = r"C:\Users\Junbo\PycharmProjects\nnUNet\nnUNet_preprocessed"

INPUT_IMAGE = r"C:\Users\Junbo\PycharmProjects\nnUNet\nnUNet_raw\Dataset001_XXY\imagesTest\image_3301523052_0000.nii.gz"
OUTPUT_SEG = r"C:\Users\Junbo\PycharmProjects\nnUNet\nnUNet_raw\Dataset001_XXY\imagesTest_output\image_3301523052.nii.gz"


def main():
    model_folder = join(
        os.environ["nnUNet_results"],
        "Dataset001_XXY/nnUNetTrainer__nnUNetPlans__3d_fullres",
    )

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", 0),
        verbose=True,
    )

    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth",
    )

    os.makedirs(os.path.dirname(OUTPUT_SEG), exist_ok=True)

    predictor.predict_from_files(
        [[INPUT_IMAGE]],
        [OUTPUT_SEG],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )

    print(f"分割结果已保存: {OUTPUT_SEG}")


if __name__ == "__main__":
    main()
