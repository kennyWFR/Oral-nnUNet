import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import ctk
import qt
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

DEFAULT_PROJECT_ROOT = r"C:\Users\Junbo\PycharmProjects\nnUNet"
DEFAULT_PYTHON_EXE = r"C:\Users\Junbo\miniconda3\envs\nnunet\python.exe"
DEFAULT_MODEL_FOLDER = (
    r"C:\Users\Junbo\PycharmProjects\nnUNet\nnUNet_results"
    r"\Dataset001_XXY\nnUNetTrainer__nnUNetPlans__3d_fullres"
)
INFERENCE_SCRIPT = (
    r"C:\Users\Junbo\PycharmProjects\nnUNet\scripts\itksnap_nnunet_predict.py"
)

# Slicer 会设置 PYTHONPATH/PYTHONHOME；若原样传给 conda python 会导致 stdlib 冲突。
_SUBPROCESS_ENV_BLOCKLIST = (
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONEXECUTABLE",
    "_PYTHON_SYSCONFIGDATA_NAME",
)


def _build_subprocess_env(project_root: Path) -> dict:
    env = os.environ.copy()
    for key in _SUBPROCESS_ENV_BLOCKLIST:
        env.pop(key, None)
    env["nnUNet_results"] = str(project_root / "nnUNet_results")
    env["nnUNet_raw"] = str(project_root / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(project_root / "nnUNet_preprocessed")
    return env


LABEL_COLORS = {
    1: (1.0, 0.0, 0.0),
    2: (0.0, 1.0, 0.0),
    3: (0.0, 0.0, 1.0),
    4: (1.0, 1.0, 0.0),
    5: (1.0, 0.0, 1.0),
    6: (0.0, 1.0, 1.0),
    7: (1.0, 0.5, 0.0),
    8: (0.5, 0.0, 1.0),
    9: (0.0, 0.5, 1.0),
    10: (0.5, 1.0, 0.0),
}


class Dataset001XXYSegmentation(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Dataset001 XXY Segmentation"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Junbo"]
        self.parent.helpText = (
            "对当前 CT 体数据运行 Dataset001_XXY nnUNet 模型，"
            "自动分割结石、腺体及其他结构。"
        )
        self.parent.acknowledgementText = (
            "基于 nnU-Net v2 与 Dataset001_XXY 训练权重。"
        )


class Dataset001XXYSegmentationWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = Dataset001XXYSegmentationLogic()

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        settingsForm = qt.QFormLayout()
        self.projectRootLineEdit = qt.QLineEdit(self.getSetting("projectRoot", DEFAULT_PROJECT_ROOT))
        self.pythonExeLineEdit = qt.QLineEdit(self.getSetting("pythonExe", DEFAULT_PYTHON_EXE))
        self.modelFolderLineEdit = qt.QLineEdit(self.getSetting("modelFolder", DEFAULT_MODEL_FOLDER))
        self.inferenceScriptLineEdit = qt.QLineEdit(
            self.getSetting("inferenceScript", INFERENCE_SCRIPT)
        )
        self.deviceComboBox = qt.QComboBox()
        self.deviceComboBox.addItems(["cuda", "cpu"])
        device = self.getSetting("device", "cuda")
        self.deviceComboBox.setCurrentText(device if device in ("cuda", "cpu") else "cuda")

        settingsForm.addRow("项目根目录:", self.projectRootLineEdit)
        settingsForm.addRow("Python 解释器:", self.pythonExeLineEdit)
        settingsForm.addRow("模型目录:", self.modelFolderLineEdit)
        settingsForm.addRow("推理脚本:", self.inferenceScriptLineEdit)
        settingsForm.addRow("设备:", self.deviceComboBox)
        self.layout.addLayout(settingsForm)

        self.layout.addWidget(qt.QLabel("输入 CT 体数据 (Scalar Volume):"))
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.layout.addWidget(self.inputSelector)

        self.layout.addWidget(qt.QLabel("输出分割 (Segmentation，可新建):"))
        self.outputSelector = slicer.qMRMLNodeComboBox()
        self.outputSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.outputSelector.noneEnabled = True
        self.outputSelector.addEnabled = True
        self.outputSelector.renameEnabled = True
        self.outputSelector.removeEnabled = True
        self.outputSelector.setMRMLScene(slicer.mrmlScene)
        self.layout.addWidget(self.outputSelector)

        self.applyButton = qt.QPushButton("运行 nnUNet 分割")
        self.applyButton.clicked.connect(self.onApplyButton)
        self.layout.addWidget(self.applyButton)

        self.statusLabel = qt.QLabel("")
        self.layout.addWidget(self.statusLabel)

        self.layout.addStretch(1)

    def getSetting(self, key, default):
        settings = qt.QSettings()
        value = settings.value(f"Dataset001XXY/{key}", default)
        return value if value is not None else default

    def saveSettings(self):
        settings = qt.QSettings()
        settings.setValue("Dataset001XXY/projectRoot", self.projectRootLineEdit.text)
        settings.setValue("Dataset001XXY/pythonExe", self.pythonExeLineEdit.text)
        settings.setValue("Dataset001XXY/modelFolder", self.modelFolderLineEdit.text)
        settings.setValue("Dataset001XXY/inferenceScript", self.inferenceScriptLineEdit.text)
        settings.setValue("Dataset001XXY/device", self.deviceComboBox.currentText)

    def onApplyButton(self):
        inputVolume = self.inputSelector.currentNode()
        if inputVolume is None:
            slicer.util.errorDisplay("请先选择一个 CT 体数据（Scalar Volume）。")
            return

        self.saveSettings()
        self.applyButton.enabled = False
        self.statusLabel.text = "正在运行 nnUNet 推理，请稍候..."
        slicer.app.processEvents()

        try:
            outputSegmentation = self.logic.run(
                inputVolumeNode=inputVolume,
                outputSegmentationNode=self.outputSelector.currentNode(),
                projectRoot=self.projectRootLineEdit.text,
                pythonExe=self.pythonExeLineEdit.text,
                modelFolder=self.modelFolderLineEdit.text,
                inferenceScript=self.inferenceScriptLineEdit.text,
                device=self.deviceComboBox.currentText,
            )
            self.outputSelector.setCurrentNode(outputSegmentation)
            self.statusLabel.text = f"完成：{outputSegmentation.GetName()}"
            slicer.util.infoDisplay("分割完成。")
        except Exception as exc:
            logging.exception("Dataset001 XXY segmentation failed")
            slicer.util.errorDisplay(f"分割失败：{exc}")
            self.statusLabel.text = "失败"
        finally:
            self.applyButton.enabled = True


class Dataset001XXYSegmentationLogic(ScriptedLoadableModuleLogic):
    def run(
        self,
        inputVolumeNode,
        outputSegmentationNode=None,
        projectRoot=DEFAULT_PROJECT_ROOT,
        pythonExe=DEFAULT_PYTHON_EXE,
        modelFolder=DEFAULT_MODEL_FOLDER,
        inferenceScript=INFERENCE_SCRIPT,
        device="cuda",
    ):
        projectRoot = Path(projectRoot)
        pythonExe = Path(pythonExe)
        modelFolder = Path(modelFolder)
        inferenceScript = Path(inferenceScript)

        self._validatePaths(pythonExe, modelFolder, inferenceScript)

        if outputSegmentationNode is None:
            outputSegmentationNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode",
                f"{inputVolumeNode.GetName()}_nnunet",
            )

        with tempfile.TemporaryDirectory(prefix="slicer_dataset001xxy_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            input_nifti = tmp_dir_path / "input_0000.nii.gz"
            output_nifti = tmp_dir_path / "input_nnunet_seg.nii.gz"

            if not slicer.util.saveNode(inputVolumeNode, str(input_nifti)):
                raise RuntimeError(f"无法导出体数据到临时文件: {input_nifti}")

            env = _build_subprocess_env(projectRoot)

            cmd = [
                str(pythonExe),
                str(inferenceScript),
                "-i",
                str(input_nifti),
                "-o",
                str(output_nifti),
                "--device",
                device,
            ]
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "外部 nnUNet 推理失败。\n"
                    f"命令: {' '.join(cmd)}\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )

            if not output_nifti.is_file():
                raise FileNotFoundError(f"未找到推理输出: {output_nifti}")

            labelVolume = slicer.util.loadLabelVolume(str(output_nifti))
            if labelVolume is None:
                raise RuntimeError(f"无法加载分割结果: {output_nifti}")

            self._importLabelVolumeToSegmentation(
                labelVolume,
                outputSegmentationNode,
                modelFolder,
                inputVolumeNode,
            )
            slicer.mrmlScene.RemoveNode(labelVolume)

        return outputSegmentationNode

    def _validatePaths(self, pythonExe, modelFolder, inferenceScript):
        missing = []
        for path in (pythonExe, modelFolder, inferenceScript):
            if not path.exists():
                missing.append(str(path))
        checkpoint = modelFolder / "fold_0" / "checkpoint_final.pth"
        if not checkpoint.is_file():
            missing.append(str(checkpoint))
        if missing:
            raise FileNotFoundError("以下路径不存在:\n" + "\n".join(missing))

    def _importLabelVolumeToSegmentation(
        self, labelVolumeNode, segmentationNode, modelFolder, referenceVolumeNode
    ):
        segmentationNode.CreateDefaultDisplayNodes()
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            labelVolumeNode,
            segmentationNode,
        )

        labelNames = self._loadLabelNames(modelFolder / "dataset.json")
        segmentation = segmentationNode.GetSegmentation()
        for i in range(segmentation.GetNumberOfSegments()):
            segment = segmentation.GetNthSegment(i)
            labelValue = int(segment.GetLabelValue())
            name = labelNames.get(labelValue, f"Label_{labelValue}")
            segment.SetName(name)
            color = LABEL_COLORS.get(labelValue, (1.0, 1.0, 1.0))
            segment.SetColor(color[0], color[1], color[2])

        displayNode = segmentationNode.GetDisplayNode()
        if displayNode:
            displayNode.SetVisibility2D(True)
            displayNode.SetVisibility3D(True)
            if hasattr(displayNode, "SetOpacity2DFill"):
                displayNode.SetOpacity2DFill(0.5)
                displayNode.SetOpacity2DOutline(0.5)
            elif hasattr(displayNode, "SetOpacity2D"):
                displayNode.SetOpacity2D(0.5)
            displayNode.SetOpacity3D(0.5)

        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(referenceVolumeNode)

    def _loadLabelNames(self, dataset_json_path):
        if not dataset_json_path.is_file():
            return {}
        with open(dataset_json_path, encoding="utf-8") as f:
            data = json.load(f)
        labels = data.get("labels", {})
        return {int(value): name for name, value in labels.items()}


class Dataset001XXYSegmentationTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_ModulePaths()

    def test_ModulePaths(self):
        self.assertTrue(Path(DEFAULT_PROJECT_ROOT).exists())
        self.delayDisplay("Dataset001XXYSegmentation path test passed", 500)


if __name__ == "__main__":
  import sys
  testing = Dataset001XXYSegmentationTest()
  testing.runTest()
