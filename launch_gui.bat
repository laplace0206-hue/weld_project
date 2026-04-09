@echo off
cd /d %~dp0
set "YOLO_CONFIG_DIR=%~dp0.ultralytics"
"%SystemRoot%\System32\cmd.exe" /c if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"
"D:\miniconda3\envs\weld_project\python.exe" scripts\weld_gui.py
pause
