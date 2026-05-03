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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_image(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")
    return img


def change_brightness(img: np.ndarray, beta: int) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)


def add_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


def add_salt_pepper_noise(img: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    n = max(1, int(amount * h * w))
    ys = rng.integers(0, h, size=n)
    xs = rng.integers(0, w, size=n)

    half = n // 2
    out[ys[:half], xs[:half]] = 255
    out[ys[half:], xs[half:]] = 0
    return out


def degrade_scale(img: np.ndarray, scale: float = 0.7) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = max(8, int(w * scale))
    new_h = max(8, int(h * scale))

    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored


def generate_variants_for_image(image_path: str, out_dir: str, seed: int = 42):
    ensure_dir(out_dir)

    img = read_image(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    rng = np.random.default_rng(seed)

    variants = {
        "orig": img,
        "dark": change_brightness(img, beta=-45),
        "bright": change_brightness(img, beta=45),
        "gauss": add_gaussian_noise(img, sigma=18.0, rng=rng),
        "sp": add_salt_pepper_noise(img, amount=0.015, rng=rng),
        "scale": degrade_scale(img, scale=0.7),
    }

    saved_files = []
    for variant_name, variant_img in variants.items():
        out_path = os.path.join(out_dir, f"{base_name}__{variant_name}.png")
        ok = cv2.imwrite(out_path, variant_img)
        if not ok:
            raise IOError(f"Не удалось сохранить файл: {out_path}")
        saved_files.append(out_path)

    return saved_files


def generate_variants_for_folder(reference_dir: str, out_dir: str, base_seed: int = 42):
    ensure_dir(out_dir)

    all_saved = []
    image_paths = list_images(reference_dir)

    for idx, image_path in enumerate(image_paths):
        saved = generate_variants_for_image(
            image_path=image_path,
            out_dir=out_dir,
            seed=base_seed + idx
        )
        all_saved.extend(saved)

    return all_saved