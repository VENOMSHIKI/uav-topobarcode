import cv2
import numpy as np


def apply_gamma(gray: np.ndarray, gamma: float = 0.75) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    corrected = np.power(gray_f, gamma)
    corrected = np.clip(corrected * 255.0, 0, 255)
    return corrected.astype(np.uint8)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def enhance_image(gray: np.ndarray, enhancement_mode: str = "gamma_then_clahe") -> np.ndarray:
    if enhancement_mode == "gamma":
        return apply_gamma(gray, gamma=0.75)
    elif enhancement_mode == "clahe":
        return apply_clahe(gray)
    elif enhancement_mode == "gamma_then_clahe":
        out = apply_gamma(gray, gamma=0.75)
        out = apply_clahe(out)
        return out
    else:
        raise ValueError(
            f"Неизвестный enhancement_mode: {enhancement_mode}. "
            f"Ожидается: gamma / clahe / gamma_then_clahe"
        )