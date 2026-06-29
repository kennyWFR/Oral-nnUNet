# 在 3D Slicer 的 Python 交互器中运行一次。
# View -> Python Interpreter:
#   exec(open(r"C:/Users/Junbo/PycharmProjects/nnUNet/scripts/install_slicer_dataset001xxy.py", encoding="utf-8").read())

import os
from pathlib import Path

import slicer

# 开发环境：指向仓库；若放在分发包 scripts/ 下则自动识别包根目录
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = os.path.normpath(
    str(PACKAGE_ROOT / "SlicerDataset001XXY" / "Dataset001XXYSegmentation")
)
OLD_WRONG_PATH = os.path.normpath(str(PACKAGE_ROOT / "SlicerDataset001XXY"))


def _normalize_paths(paths):
    if paths is None:
        return []
    if isinstance(paths, str):
        return [paths] if paths else []
    return list(paths)


def _get_current_paths():
    settings = slicer.app.revisionUserSettings()
    paths = settings.value("Modules/AdditionalPaths")
    if paths is None and hasattr(slicer.util, "settingsValue"):
        try:
            paths = slicer.util.settingsValue("Modules/AdditionalPaths", [])
        except Exception:
            paths = []
    return _normalize_paths(paths)


def _save_paths(paths):
    settings = slicer.app.revisionUserSettings()
    settings.setValue("Modules/AdditionalPaths", paths)
    settings.sync()
    user_settings = slicer.app.userSettings()
    if user_settings is not None:
        user_settings.setValue("Modules/AdditionalPaths", paths)
        user_settings.sync()
    if hasattr(slicer.util, "setSettingsValue"):
        try:
            slicer.util.setSettingsValue("Modules/AdditionalPaths", paths)
        except Exception:
            pass


def _norm_set(paths):
    return {os.path.normpath(p) for p in paths if p}


print(f"项目/分发包根目录: {PACKAGE_ROOT}")
print(f"模块路径: {MODULE_PATH}")

if not os.path.isdir(MODULE_PATH):
    raise FileNotFoundError(f"未找到模块目录: {MODULE_PATH}")

current_paths = _get_current_paths()
changed = False

if OLD_WRONG_PATH in _norm_set(current_paths):
    current_paths = [p for p in current_paths if os.path.normpath(p) != OLD_WRONG_PATH]
    changed = True
    print(f"已移除错误路径: {OLD_WRONG_PATH}")

if MODULE_PATH not in _norm_set(current_paths):
    current_paths.append(MODULE_PATH)
    changed = True
    print(f"已添加正确路径: {MODULE_PATH}")
else:
    print(f"正确路径已存在: {MODULE_PATH}")

if changed:
    _save_paths(current_paths)

module_manager = slicer.app.moduleManager()
if hasattr(module_manager, "addModulePaths"):
    module_manager.addModulePaths([MODULE_PATH])

print("\n请完全退出并重启 3D Slicer。")
print("重启后在模块搜索框输入: Dataset001 / XXY")
print("模块全名: Dataset001 XXY Segmentation")
print(f"\n当前 AdditionalPaths:\n{current_paths}")
