import argparse
import os
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_image_unicode(path: str):
    """
    Чтение изображения через np.fromfile + cv2.imdecode.
    На Windows это надежнее, чем обычный cv2.imread,
    если в пути есть русские буквы или спецсимволы.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except Exception:
        return None

    if data.size == 0:
        return None

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def iter_images(folder: str):
    images = []
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            ext = Path(name).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                images.append(os.path.join(root, name))
    return images


def choose_image_from_dir(folder: str):
    images = iter_images(folder)

    if not images:
        raise FileNotFoundError(f"В папке нет изображений: {folder}")

    print("\nНайденные изображения:")
    for i, path in enumerate(images, start=1):
        print(f"{i}. {path}")

    print("0. Выход")

    while True:
        value = input("\nВведи номер изображения для нарезки ROI: ").strip().lower()

        if value in {"0", "q", "й", "exit"}:
            return None

        try:
            index = int(value)
            if 1 <= index <= len(images):
                return images[index - 1]
        except ValueError:
            pass

        print("Неверный номер. Попробуй еще раз.")


def next_index(out_dir: str, prefix: str):
    ensure_dir(out_dir)
    nums = []

    for name in os.listdir(out_dir):
        if not name.lower().endswith(".png"):
            continue

        stem = os.path.splitext(name)[0]
        if not stem.startswith(prefix + "_"):
            continue

        tail = stem.replace(prefix + "_", "")
        try:
            nums.append(int(tail))
        except ValueError:
            pass

    return max(nums, default=0) + 1


def resize_for_screen(img, max_width=1200, max_height=800):
    h, w = img.shape[:2]

    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(1.0, scale_w, scale_h)

    if scale >= 1.0:
        return img.copy(), 1.0

    new_w = int(w * scale)
    new_h = int(h * scale)

    display = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return display, scale


def make_square_box(x0, y0, x1, y1, img_w, img_h):
    """
    Превращает выделенный прямоугольник в квадрат.
    Квадрат строится вокруг центра выделения.
    Это защищает ROI от искажения при resize в 128x128.
    """
    w = x1 - x0
    h = y1 - y0

    side = max(w, h)

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    sx0 = cx - side // 2
    sy0 = cy - side // 2
    sx1 = sx0 + side
    sy1 = sy0 + side

    if sx0 < 0:
        sx1 -= sx0
        sx0 = 0

    if sy0 < 0:
        sy1 -= sy0
        sy0 = 0

    if sx1 > img_w:
        shift = sx1 - img_w
        sx0 -= shift
        sx1 = img_w

    if sy1 > img_h:
        shift = sy1 - img_h
        sy0 -= shift
        sy1 = img_h

    sx0 = max(0, sx0)
    sy0 = max(0, sy0)
    sx1 = min(img_w, sx1)
    sy1 = min(img_h, sy1)

    return sx0, sy0, sx1, sy1


def save_rois(img, rois, scale, out_dir, prefix, size, force_square=True):
    idx = next_index(out_dir, prefix)
    saved = 0

    img_h, img_w = img.shape[:2]

    for roi in rois:
        x, y, w, h = roi

        if w == 0 or h == 0:
            continue

        # Пересчет координат с уменьшенной картинки на оригинал
        x0 = int(x / scale)
        y0 = int(y / scale)
        x1 = int((x + w) / scale)
        y1 = int((y + h) / scale)

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(img_w, x1)
        y1 = min(img_h, y1)

        if force_square:
            x0, y0, x1, y1 = make_square_box(x0, y0, x1, y1, img_w, img_h)

        crop = img[y0:y1, x0:x1]

        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

        out_name = f"{prefix}_{idx:02d}.png"
        out_path = os.path.join(out_dir, out_name)

        ok = cv2.imwrite(out_path, crop_resized)
        if not ok:
            print(f"ERROR: не удалось сохранить {out_path}")
            continue

        print(
            f"Сохранено: {out_path} | "
            f"x={x0}, y={y0}, w={x1 - x0}, h={y1 - y0}, size={size}x{size}"
        )

        idx += 1
        saved += 1

    return saved


def crop_one_image(image_path, out_dir, prefix, size, force_square):
    img = read_image_unicode(image_path)
    if img is None:
        print(f"ERROR: не удалось открыть изображение: {image_path}")
        return

    ensure_dir(out_dir)

    display, scale = resize_for_screen(img)

    print(f"\nОткрыто изображение: {image_path}")
    print("Выделяй ROI мышкой.")
    print("SPACE или ENTER — подтвердить текущий ROI.")
    print("ESC — закончить выбор и вернуться к списку.")
    print("C — отменить текущий выбор.")
    print("Совет: выделяй почти квадратную область вокруг главной структуры.")

    rois = cv2.selectROIs(
        "Select multiple ROIs",
        display,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyAllWindows()

    if len(rois) == 0:
        print("ROI не выбраны.")
        return

    saved = save_rois(
        img=img,
        rois=rois,
        scale=scale,
        out_dir=out_dir,
        prefix=prefix,
        size=size,
        force_square=force_square
    )

    print(f"Сохранено ROI: {saved}")


def main():
    parser = argparse.ArgumentParser(description="Удобная нарезка ROI из папки или одного изображения")
    parser.add_argument("--image", default=None, help="Путь к одному изображению")
    parser.add_argument("--image-dir", default=None, help="Папка с изображениями")
    parser.add_argument("--out-dir", required=True, help="Куда сохранять ROI")
    parser.add_argument("--prefix", required=True, help="Класс: bridge, river, street, roof, city_block")
    parser.add_argument("--size", type=int, default=128, help="Размер итогового ROI")
    parser.add_argument(
        "--no-square",
        action="store_true",
        help="Не расширять выделение до квадрата"
    )

    args = parser.parse_args()

    force_square = not args.no_square

    if args.image is None and args.image_dir is None:
        raise ValueError("Нужно указать --image или --image-dir")

    if args.image is not None:
        crop_one_image(
            image_path=args.image,
            out_dir=args.out_dir,
            prefix=args.prefix,
            size=args.size,
            force_square=force_square
        )
        return

    while True:
        image_path = choose_image_from_dir(args.image_dir)

        if image_path is None:
            print("Выход из нарезки.")
            break

        crop_one_image(
            image_path=image_path,
            out_dir=args.out_dir,
            prefix=args.prefix,
            size=args.size,
            force_square=force_square
        )


if __name__ == "__main__":
    main()