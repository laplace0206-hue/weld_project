from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def load_bgr_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    return image


def gray_world_balance(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mean_b, mean_g, mean_r = image.mean(axis=(0, 1))
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    scales = np.array([
        mean_gray / max(mean_b, 1e-6),
        mean_gray / max(mean_g, 1e-6),
        mean_gray / max(mean_r, 1e-6),
    ], dtype=np.float32)
    balanced = np.clip(image * scales, 0, 255)
    return balanced.astype(np.uint8)


def clahe_enhance(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def traditional_preprocess(image: np.ndarray) -> np.ndarray:
    balanced = gray_world_balance(image)
    enhanced = clahe_enhance(balanced)
    denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=35, sigmaSpace=35)
    return denoised


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    return tensor
