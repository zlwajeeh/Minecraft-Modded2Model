@echo off
rem Build dist\Minecraft-Modded2Model-v<version>.exe (needs: python -m pip install --user pyinstaller numpy scipy pillow)
cd /d "%~dp0.."
python packaging\build_exe.py
