"""Build dist/Minecraft-Modded2Model-v<version>.exe with PyInstaller. The version comes from
modded2model/__init__.py and goes in the file name and the exe's Properties > Details."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from modded2model import __version__  # noqa: E402

HERE = os.path.join(ROOT, "packaging")
name = f"Minecraft-Modded2Model-v{__version__}"
nums = tuple(int(x) for x in (__version__.split(".") + ["0"] * 4)[:4])
info = f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={nums}, prodvers={nums}),
  kids=[StringFileInfo([StringTable('040904B0', [
    StringStruct('ProductName', 'Minecraft-Modded2Model'),
    StringStruct('FileDescription', 'Minecraft-Modded2Model v{__version__}'),
    StringStruct('ProductVersion', '{__version__}'),
    StringStruct('FileVersion', '{__version__}'),
    StringStruct('OriginalFilename', '{name}.exe'),
    StringStruct('InternalName', 'Minecraft-Modded2Model')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])])
"""
build = os.path.join(ROOT, "build")
os.makedirs(build, exist_ok=True)
version_file = os.path.join(build, "version_info.txt")
with open(version_file, "w", encoding="utf-8") as f:
    f.write(info)
subprocess.check_call([sys.executable, os.path.join(HERE, "make_icon.py")])
subprocess.check_call([
    sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
    "--name", name, "--icon", os.path.join(HERE, "icon.ico"), "--version-file", version_file,
    "--collect-submodules", "modded2model",
    "--exclude-module", "matplotlib", "--exclude-module", "pandas",
    "--exclude-module", "IPython", "--exclude-module", "pytest",
    os.path.join(HERE, "launcher.py")], cwd=ROOT)
print(f"Built dist/{name}.exe")
