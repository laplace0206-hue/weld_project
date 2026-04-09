@echo off
cd /d %~dp0
"D:\miniconda\envs\machine\python.exe" scripts\train_custom_yolo.py --config configs\experiment.yaml --epochs 100 --batch 2 --imgsz 640 --workers 0 --device 0 --name formal_custom_yolo_100e
pause
