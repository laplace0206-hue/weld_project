# Weld Project

面向焊缝缺陷检测研究的代码框架，围绕以下目标组织：

- 复现 `YOLOv8` 基线并完成训练、验证、导出
- 提供面向论文/课题的改进模块骨架：自适应预处理、改进 FPN、轻量注意力、损失函数优化
- 支持对当前工作区中的 `1` 数据集进行统计分析与路径修正

## 目录结构

- `configs/`：数据集与实验配置
- `scripts/`：数据分析、训练、评估、导出入口脚本
- `src/weld_project/`：核心 Python 包

## 当前数据集

默认使用工作区中的 `../1` 数据集：

- 类别数：4
- 类别名：`Crack`、`Porosity`、`Spatters`、`Welding line`
- 标注格式：YOLO detection

## 快速开始

1. 安装依赖

```powershell
pip install -r requirements.txt
```

2. 分析数据集

```powershell
python scripts/analyze_dataset.py --dataset ..\1
```

3. 训练基线 YOLOv8

```powershell
python scripts/train.py --config configs/experiment.yaml --stage baseline
```

4. 训练改进实验（框架入口）

```powershell
python scripts/train.py --config configs/experiment.yaml --stage full
```

5. 评估

```powershell
python scripts/evaluate.py --config configs/experiment.yaml
```

6. 单张图片缺陷识别与文字分析

```powershell
python scripts/predict_report.py --source ..\1\test\images\bad_weld_vid163_jpeg_jpg.rf.b385ff7827589632ebfea01a9f5520fd.jpg --device 0
```

脚本会优先自动使用最新训练生成的 `best.pt`；如果没有，则回退到工作区中的 `yolo26n.pt`。您也可以手动传入 `--weights` 指定权重文件。

7. 图形界面识别

```powershell
python scripts/weld_gui.py
```

在 Windows 下也可以直接双击 [weld_project/launch_gui.bat](weld_project/launch_gui.bat) 启动界面。界面支持：

- 选择焊缝图片
- 选择整批图片所在文件夹
- 可选选择权重文件
- 原图与识别结果图并排显示
- 输出标注后的缺陷结果图
- 输出文字分析结果
- 批量输出 CSV、TXT、JSON 汇总

非代码用户可按下面流程直接使用 GUI：

1. 双击 [weld_project/launch_gui.bat](weld_project/launch_gui.bat)。
2. 单张识别时点击“选择图片”，批量识别时点击“选择文件夹”。
3. 如果界面里已经自动填好权重路径，通常可以直接使用；否则点击“选择权重”手动选择 [weld_project/runs/weld_defect_research/smoke_custom_yolo_bg2/weights/best.pt](weld_project/runs/weld_defect_research/smoke_custom_yolo_bg2/weights/best.pt) 或正式训练得到的 best.pt。
4. 点击“单张识别”或“批量识别”。
5. 结果默认保存在 [weld_project/outputs/gui_predict](weld_project/outputs/gui_predict)；批量汇总文件保存在 [weld_project/outputs/gui_predict/batch](weld_project/outputs/gui_predict/batch)。

8. 正式训练快捷启动

可直接双击以下脚本启动正式训练：

- [weld_project/train_formal_50epochs.bat](weld_project/train_formal_50epochs.bat)
- [weld_project/train_formal_100epochs.bat](weld_project/train_formal_100epochs.bat)

当前已经完成过一轮 1 epoch 冒烟训练，可直接使用其权重进行推理：

- [weld_project/runs/weld_defect_research/smoke_custom_yolo_bg2/weights/best.pt](weld_project/runs/weld_defect_research/smoke_custom_yolo_bg2/weights/best.pt)

9. 验证损失函数是否真正接入训练链路

```powershell
python scripts/verify_loss_chain.py --loss-type focal_eiou
```

如果输出 `CustomV8DetectionLoss` 和 `CustomBboxLoss`，说明损失优化已经真正进入 Ultralytics 训练链路，而不是只停留在结构改动。

10. 汇总训练结果，便于论文写表格

```powershell
python scripts/summarize_run.py runs/weld_defect_research/formal_custom_yolo_50e runs/weld_defect_research/loss_chain_verify_feiou_fast
```

该脚本会输出每个 run 的最佳 epoch、mAP50、mAP50-95、precision、recall。

## 研究建议

推荐按如下消融顺序进行：

1. `baseline`：原始 YOLOv8
2. `preprocess`：加入自适应预处理
3. `neck`：替换为增强型 FPN
4. `attention`：增加轻量注意力
5. `full`：叠加损失函数优化与全模块组合

## 说明

- 本工程优先保证**课题研究流程完整**与**代码框架清晰**。
- 基线训练与验证直接基于 `ultralytics`。
- 改进模块采用独立 PyTorch 实现，便于后续移植到 YOLOv8/RT-DETR 或自定义检测器中。
- `predict_report.py` 会输出三类结果：标注后的图片、文字分析报告、结构化 JSON 报告。
- 如果你是初学者，建议先阅读 `项目学习说明_新手版.md`。
