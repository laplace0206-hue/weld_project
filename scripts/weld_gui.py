from __future__ import annotations

import contextlib
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import subprocess

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageTk

from src.weld_project.inference import format_text_report
from src.weld_project.inference.predictor import resolve_default_weights, run_batch_inference, run_image_inference


class WeldInspectorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("焊缝缺陷识别")
        self.root.geometry("1200x760")

        self.image_path_var = tk.StringVar()
        self.folder_path_var = tk.StringVar()
        self.weights_var = tk.StringVar()
        self.status_var = tk.StringVar(value="请选择一张焊缝图片")
        self.conf_var = tk.StringVar(value="0.25")
        self.imgsz_var = tk.StringVar(value="640")
        self.device_var = tk.StringVar(value="0")
        self.preprocess_var = tk.BooleanVar(value=False)
        self.output_dir = PROJECT_ROOT / "outputs" / "gui_predict"
        self.current_input_photo: ImageTk.PhotoImage | None = None
        self.current_result_photo: ImageTk.PhotoImage | None = None

        default_weights = resolve_default_weights(PROJECT_ROOT)
        if default_weights is not None:
            self.weights_var.set(str(default_weights))

        self._build_layout()

    def _build_layout(self) -> None:
        control = ttk.Frame(self.root, padding=12)
        control.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control, text="图片路径").grid(row=0, column=0, sticky="w")
        ttk.Entry(control, textvariable=self.image_path_var, width=90).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(control, text="选择图片", command=self.choose_image).grid(row=0, column=2, padx=6)

        ttk.Label(control, text="批量文件夹").grid(row=1, column=0, sticky="w")
        ttk.Entry(control, textvariable=self.folder_path_var, width=90).grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(control, text="选择文件夹", command=self.choose_folder).grid(row=1, column=2, padx=6)

        ttk.Label(control, text="权重路径").grid(row=2, column=0, sticky="w")
        ttk.Entry(control, textvariable=self.weights_var, width=90).grid(row=2, column=1, sticky="ew", padx=6)
        ttk.Button(control, text="选择权重", command=self.choose_weights).grid(row=2, column=2, padx=6)

        ttk.Label(control, text="conf").grid(row=3, column=0, sticky="w")
        ttk.Entry(control, textvariable=self.conf_var, width=12).grid(row=3, column=1, sticky="w", padx=6)
        ttk.Label(control, text="imgsz").grid(row=3, column=1, sticky="w", padx=(140, 6))
        ttk.Entry(control, textvariable=self.imgsz_var, width=12).grid(row=3, column=1, sticky="w", padx=(190, 6))
        ttk.Label(control, text="device").grid(row=3, column=1, sticky="w", padx=(300, 6))
        ttk.Entry(control, textvariable=self.device_var, width=12).grid(row=3, column=1, sticky="w", padx=(350, 6))
        ttk.Checkbutton(control, text="传统预处理", variable=self.preprocess_var).grid(row=3, column=1, sticky="w", padx=(470, 6))
        ttk.Button(control, text="单张识别", command=self.run_inference).grid(row=3, column=2, padx=6)
        ttk.Button(control, text="批量识别", command=self.run_batch_inference).grid(row=3, column=3, padx=6)
        ttk.Button(control, text="打开结果目录", command=self.open_output_dir).grid(row=3, column=4, padx=6)

        control.columnconfigure(1, weight=1)

        content = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        left = ttk.Frame(content, padding=8)
        right = ttk.Frame(content, padding=8)
        content.add(left, weight=3)
        content.add(right, weight=2)

        image_panel = ttk.Panedwindow(left, orient=tk.HORIZONTAL)
        image_panel.pack(fill=tk.BOTH, expand=True)

        input_frame = ttk.Frame(image_panel, padding=6)
        result_frame = ttk.Frame(image_panel, padding=6)
        image_panel.add(input_frame, weight=1)
        image_panel.add(result_frame, weight=1)

        ttk.Label(input_frame, text="原始图片").pack(anchor="w")
        self.input_image_label = ttk.Label(input_frame, text="原图将在这里显示", anchor="center")
        self.input_image_label.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        ttk.Label(result_frame, text="识别结果图").pack(anchor="w")
        self.result_image_label = ttk.Label(result_frame, text="标注结果将在这里显示", anchor="center")
        self.result_image_label.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        ttk.Label(right, text="文字分析报告").pack(anchor="w")
        self.report_text = tk.Text(right, wrap=tk.WORD, font=("Microsoft YaHei UI", 11))
        self.report_text.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择焊缝图片",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )
        if path:
            self.image_path_var.set(path)

    def choose_weights(self) -> None:
        path = filedialog.askopenfilename(
            title="选择模型权重",
            filetypes=[("PyTorch Weights", "*.pt")],
        )
        if path:
            self.weights_var.set(path)

    def choose_folder(self) -> None:
        path = filedialog.askdirectory(title="选择待批量识别的图片文件夹")
        if path:
            self.folder_path_var.set(path)

    def run_inference(self) -> None:
        image_path = self.image_path_var.get().strip()
        weights_path = self.weights_var.get().strip() or None
        if not image_path:
            messagebox.showerror("缺少输入", "请先选择焊缝图片。")
            return
        if weights_path is None:
            resolved = resolve_default_weights(PROJECT_ROOT)
            if resolved is None:
                messagebox.showerror("缺少权重", "未找到可用权重，请先选择权重文件。")
                return
            weights_path = str(resolved)
            self.weights_var.set(weights_path)

        try:
            self.status_var.set("正在识别，请稍候...")
            self.root.update_idletasks()
            self._show_input_image(Path(image_path))
            result = run_image_inference(
                weights_path=weights_path,
                source_path=image_path,
                output_dir=self.output_dir,
                imgsz=int(self.imgsz_var.get()),
                conf=float(self.conf_var.get()),
                device=self.device_var.get().strip(),
                preprocess=self.preprocess_var.get(),
            )
            annotated = result["output_paths"]["annotated_image"]
            self._show_result_image(annotated)
            self.report_text.delete("1.0", tk.END)
            self.report_text.insert(tk.END, format_text_report(result["summary"]))
            self.status_var.set(f"识别完成，结果已保存到: {self.output_dir}")
        except Exception as exc:
            self.status_var.set("识别失败")
            messagebox.showerror("识别失败", str(exc))

    def run_batch_inference(self) -> None:
        folder_path = self.folder_path_var.get().strip()
        weights_path = self.weights_var.get().strip() or None
        if not folder_path:
            messagebox.showerror("缺少输入", "请先选择图片文件夹。")
            return
        if weights_path is None:
            resolved = resolve_default_weights(PROJECT_ROOT)
            if resolved is None:
                messagebox.showerror("缺少权重", "未找到可用权重，请先选择权重文件。")
                return
            weights_path = str(resolved)
            self.weights_var.set(weights_path)

        try:
            self.status_var.set("正在批量识别，请稍候...")
            self.root.update_idletasks()
            batch_dir = self.output_dir / "batch"
            result = run_batch_inference(
                weights_path=weights_path,
                source_dir=folder_path,
                output_dir=batch_dir,
                imgsz=int(self.imgsz_var.get()),
                conf=float(self.conf_var.get()),
                device=self.device_var.get().strip(),
                preprocess=self.preprocess_var.get(),
            )
            image_results = result["summary"]["image_results"]
            if image_results:
                first_item = image_results[0]
                self._show_input_image(Path(folder_path) / first_item["image_name"])
                self._show_result_image(Path(first_item["annotated_image"]))
            self.report_text.delete("1.0", tk.END)
            self.report_text.insert(tk.END, self._format_batch_summary_text(result["summary"], result["output_paths"]))
            self.status_var.set(f"批量识别完成，结果已保存到: {batch_dir}")
        except Exception as exc:
            self.status_var.set("批量识别失败")
            messagebox.showerror("批量识别失败", str(exc))

    def _show_input_image(self, image_path: Path) -> None:
        image = Image.open(image_path)
        image.thumbnail((500, 640))
        self.current_input_photo = ImageTk.PhotoImage(image)
        self.input_image_label.configure(image=self.current_input_photo, text="")

    def _show_result_image(self, image_path: Path) -> None:
        image = Image.open(image_path)
        image.thumbnail((500, 640))
        self.current_result_photo = ImageTk.PhotoImage(image)
        self.result_image_label.configure(image=self.current_result_photo, text="")

    def open_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(self.output_dir)])

    def _format_batch_summary_text(self, summary: dict[str, object], output_paths: dict[str, Path]) -> str:
        class_counter = summary.get("class_counter", {})
        lines = [
            f"输入文件夹: {summary['source_dir']}",
            f"图片总数: {summary['total_images']}",
            f"检测到缺陷的图片数: {summary['images_with_defects']}",
            f"未检测到缺陷的图片数: {summary['images_without_defects']}",
            "",
            "类别总计:",
        ]
        if class_counter:
            for class_name, count in sorted(class_counter.items(), key=lambda item: item[1], reverse=True):
                lines.append(f"- {class_name}: {count}")
        else:
            lines.append("- 未检测到任何缺陷")

        lines.extend(
            [
                "",
                f"CSV汇总: {output_paths['csv_report']}",
                f"文字汇总: {output_paths['text_report']}",
                f"JSON汇总: {output_paths['json_report']}",
            ]
        )
        return "\n".join(lines)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    with contextlib.suppress(Exception):
        style.theme_use("clam")
    app = WeldInspectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
