import os
import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: str):
    files = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        ext = os.path.splitext(name.lower())[1]
        if os.path.isfile(path) and ext in IMAGE_EXTENSIONS:
            files.append(path)
    return files


def read_base_steps(image_path: str, size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    mean_brightness_raw = float(np.mean(gray_resized))
    median_raw = float(np.median(gray_resized))
    std_raw = float(np.std(gray_resized))
    p10_raw = float(np.percentile(gray_resized, 10))
    p90_raw = float(np.percentile(gray_resized, 90))

    gray_norm = cv2.normalize(gray_resized, None, 0, 255, cv2.NORM_MINMAX)

    return {
        "original_bgr": img,
        "gray": gray,
        "gray_resized": gray_resized,
        "gray_norm": gray_norm,
        "mean_brightness_raw": mean_brightness_raw,
        "median_raw": median_raw,
        "std_raw": std_raw,
        "p10_raw": p10_raw,
        "p90_raw": p90_raw,
    }


def apply_blur(gray: np.ndarray, blur_ksize: int = 3) -> np.ndarray:
    if blur_ksize and blur_ksize > 1:
        if blur_ksize % 2 == 0:
            raise ValueError("blur_ksize должен быть нечетным: 3, 5, 7 ...")
        return cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return gray.copy()