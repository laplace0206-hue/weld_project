@echo off
cd /d %~dp0
"D:\miniconda\envs\machine\python.exe" scripts\monitor_training.py --run-dir runs\weld_defect_research\formal_custom_yolo_50e --interval 5
pause
