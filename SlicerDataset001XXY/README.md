# 3D Slicer 扩展：Dataset001 XXY Segmentation

对 CT 体数据运行 `Dataset001_XXY` nnUNet 模型（结石、腺体等），并在 Slicer 中显示分割结果。

## 方式 A：不编译，直接加载（推荐）

1. 在 Slicer 的 **Python 交互器** 中执行（推荐）：

   ```python
   exec(open(r"C:/Users/Junbo/PycharmProjects/nnUNet/scripts/install_slicer_dataset001xxy.py", encoding="utf-8").read())
   ```

   或双击运行（备选）：

   `scripts/register_slicer_module_path.bat`

2. 完全退出并重新打开 **3D Slicer**

3. 在模块搜索框输入 **Dataset001**，打开 **Dataset001 XXY Segmentation**

若上述方式未生效，手动添加模块路径：

- **Edit → Settings → Modules → Additional module paths**
- 添加：`C:\Users\Junbo\PycharmProjects\nnUNet\SlicerDataset001XXY\Dataset001XXYSegmentation`

  **注意：必须是 `Dataset001XXYSegmentation` 这一层，不是上一级 `SlicerDataset001XXY`。**

## 方式 B：作为扩展编译安装（可选）

需要已安装 3D Slicer 开发环境，在 Slicer 源码树外配置：

```powershell
cmake -G "Visual Studio 17 2022" -A x64 ^
  -DEXTENSION_NAME=SlicerDataset001XXY ^
  -DSlicer_DIR="C:\Path\To\Slicer-build" ^
  -DCMAKE_BUILD_TYPE=Release ^
  "C:\Users\Junbo\PycharmProjects\nnUNet\SlicerDataset001XXY"

cmake --build . --config Release
```

## 使用步骤

1. 在 Slicer 中加载 CT（**DICOM 导入** 或 **Load Data**）
2. 打开模块 **Dataset001 XXY Segmentation**
3. 确认设置（首次使用检查以下路径）：
   - **Python 解释器**：`C:\Users\Junbo\miniconda3\envs\nnunet\python.exe`
   - **项目根目录**：`C:\Users\Junbo\PycharmProjects\nnUNet`
   - **模型目录**：`...\nnUNet_results\Dataset001_XXY\nnUNetTrainer__nnUNetPlans__3d_fullres`
4. 选择 **Input Volume**（当前 CT）
5. 点击 **运行 nnUNet 分割**
6. 结果出现在 **Segmentation** 节点中，可在 Segment Editor 中编辑

## 原理

- Slicer 模块负责 UI 与结果导入
- 实际推理在 **外部 conda 环境** 中调用 `scripts/itksnap_nnunet_predict.py`
- 避免在 Slicer 内置 Python 中重复安装 PyTorch/CUDA

## 标签含义

| 值 | 名称 |
|----|------|
| 1 | stone |
| 2 | bland（腺体） |
| 3–10 | 颅骨、牙齿、眼耳等 |

## 故障排查

| 问题 | 处理 |
|------|------|
| 找不到模块 | 检查 Additional module paths，重启 Slicer |
| 分割失败 | 检查 Python 路径、模型 `checkpoint_final.pth` 是否存在 |
| CUDA 报错 | 模块中将设备改为 `cpu`（很慢） |
| 结果方向不对 | 确认 CT 与训练数据同为 NIfTI，必要时用 Resample 模块 |
