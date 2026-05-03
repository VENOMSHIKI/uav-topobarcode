import cv2
import numpy as np


def apply_gamma(gray: np.ndarray, gamma: float = 0.75) -> np.ndarray:
    """
    Gamma correction.
    gamma < 1.0 делает темное изображение светлее.
    """
    gray_f = gray.astype(np.float32) / 255.0
    corrected = np.power(gray_f, gamma)
    corrected = np.clip(corrected * 255.0, 0, 255)
    return corrected.astype(np.uint8)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    CLAHE для локального улучшения контраста.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def enhance_image(gray: np.ndarray, enhancement_mode: str = "gamma_then_clahe") -> np.ndarray:
    """
    enhancement_mode:
    - gamma
    - clahe
    - gamma_then_clahe
    """
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


def preprocess_steps(
    image_path: str,
    size=(128, 128),
    blur_ksize: int = 3,
    preprocess_mode: str = "none",   # none / always / auto
    enhancement_mode: str = "gamma_then_clahe",
    dark_threshold: float = 145.0,
):
    """
    preprocess_mode:
    - none   : только baseline
    - always : улучшение всегда
    - auto   : улучшение только если ROI темный

    dark_threshold применяется к gray_resized ДО нормализации.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    # ВАЖНО: яркость оцениваем до normalize
    mean_brightness_raw = float(np.mean(gray_resized))
    is_dark = mean_brightness_raw < dark_threshold

    gray_norm = cv2.normalize(gray_resized, None, 0, 255, cv2.NORM_MINMAX)

    gray_enhanced = gray_norm.copy()
    enhancement_applied = False

    if preprocess_mode == "none":
        gray_enhanced = gray_norm.copy()

    elif preprocess_mode == "always":
        gray_enhanced = enhance_image(gray_norm, enhancement_mode=enhancement_mode)
        enhancement_applied = True

    elif preprocess_mode == "auto":
        if is_dark:
            gray_enhanced = enhance_image(gray_norm, enhancement_mode=enhancement_mode)
            enhancement_applied = True
        else:
            gray_enhanced = gray_norm.copy()

    else:
        raise ValueError(
            f"Неизвестный preprocess_mode: {preprocess_mode}. "
            f"Ожидается: none / always / auto"
        )

    if blur_ksize and blur_ksize > 1:
        if blur_ksize % 2 == 0:
            raise ValueError("blur_ksize должен быть нечетным: 3, 5, 7 ...")
        gray_final = cv2.GaussianBlur(gray_enhanced, (blur_ksize, blur_ksize), 0)
    else:
        gray_final = gray_enhanced.copy()

    return {
        "original_bgr": img,
        "gray": gray,
        "gray_resized": gray_resized,
        "gray_norm": gray_norm,
        "gray_enhanced": gray_enhanced,
        "gray_final": gray_final,
        "mean_brightness_raw": mean_brightness_raw,
        "is_dark": is_dark,
        "enhancement_applied": enhancement_applied,
        "preprocess_mode": preprocess_mode,
        "enhancement_mode": enhancement_mode,
        "dark_threshold": dark_threshold,
    }


def load_roi(
    image_path: str,
    size=(128, 128),
    blur_ksize: int = 3,
    preprocess_mode: str = "none",
    enhancement_mode: str = "gamma_then_clahe",
    dark_threshold: float = 145.0,
) -> np.ndarray:
    steps = preprocess_steps(
        image_path=image_path,
        size=size,
        blur_ksize=blur_ksize,
        preprocess_mode=preprocess_mode,
        enhancement_mode=enhancement_mode,
        dark_threshold=dark_threshold,
    )
    return steps["gray_final"]