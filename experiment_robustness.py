import argparse
import csv
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from src.database import match_against_database
from src.preprocess import load_roi
from src.barcode import build_topo_barcode, barcode_to_vector


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: str):
    """
    Возвращает список изображений из папки и вложенных подпапок.
    Нужно для эксперимента TheWorld50.
    """
    files = []

    if not os.path.isdir(folder):
        return files

    for root, dirs, names in os.walk(folder):
        for name in sorted(names):
            path = os.path.join(root, name)
            ext = os.path.splitext(name.lower())[1]

            if os.path.isfile(path) and ext in IMAGE_EXTENSIONS:
                files.append(path)

    return sorted(files)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def true_name_from_query(filename: str):
    """
    bridge_08__bright.png -> bridge_08
    bridge_08.png -> bridge_08
    """
    stem = Path(filename).stem

    if "__" in stem:
        return stem.split("__")[0]

    return stem


def class_from_name(name: str):
    """
    bridge_08 -> bridge
    field_mark_01 -> field_mark
    """
    parts = name.split("_")

    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])

    return parts[0]


def variant_from_query(filename: str):
    """
    bridge_08__bright.png -> bright
    bridge_08.png -> orig
    """
    stem = Path(filename).stem

    if "__" in stem:
        return stem.split("__", 1)[1]

    return "orig"


def resize_keep_size(img, scale_factor: float):
    """
    Имитирует масштабирование:
    сначала уменьшаем/увеличиваем, потом возвращаем к исходному размеру.
    """
    h, w = img.shape[:2]

    new_w = max(8, int(w * scale_factor))
    new_h = max(8, int(h * scale_factor))

    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    return restored


def add_gaussian_noise(img, sigma: float = 16.0):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_pepper(img, amount: float = 0.02):
    out = img.copy()

    h, w = img.shape[:2]
    n = int(h * w * amount)

    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)

    half = n // 2

    if img.ndim == 2:
        out[ys[:half], xs[:half]] = 0
        out[ys[half:], xs[half:]] = 255
    else:
        out[ys[:half], xs[:half], :] = 0
        out[ys[half:], xs[half:], :] = 255

    return out


def make_bright(img, beta: int = 45):
    out = img.astype(np.int16) + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def make_dark(img, beta: int = -45):
    out = img.astype(np.int16) + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def generate_queries(ref_dir: str, query_dir: str):
    """
    Генерирует искаженные query по эталонным ROI.

    Для каждого reference:
    - orig;
    - bright;
    - dark;
    - gauss;
    - sp;
    - scale.

    Если файлы уже есть — папка очищается и создается заново.
    """
    ensure_dir(query_dir)

    for old in list_images(query_dir):
        os.remove(old)

    ref_paths = list_images(ref_dir)

    if not ref_paths:
        raise FileNotFoundError(f"Нет изображений в ref-dir: {ref_dir}")

    generated = []

    for ref_path in ref_paths:
        name = Path(ref_path).stem
        img = cv2.imread(ref_path, cv2.IMREAD_COLOR)

        if img is None:
            print(f"SKIP: не удалось открыть {ref_path}")
            continue

        variants = {
            "orig": img,
            "bright": make_bright(img, beta=45),
            "dark": make_dark(img, beta=-45),
            "gauss": add_gaussian_noise(img, sigma=16.0),
            "sp": add_salt_pepper(img, amount=0.02),
            "scale": resize_keep_size(img, scale_factor=0.75),
        }

        for variant_name, variant_img in variants.items():
            out_name = f"{name}__{variant_name}.png"
            out_path = os.path.join(query_dir, out_name)

            cv2.imwrite(out_path, variant_img)
            generated.append(out_path)

    return generated


def measure_descriptor_time_ms(
    image_path: str,
    preprocess_mode: str,
    enhancement_mode: str,
    mean_threshold: float,
    p90_threshold: float,
    threshold_mode: str,
):
    """
    Замеряет только построение дескриптора:
    load_roi -> build_topo_barcode -> barcode_to_vector.
    """
    t0 = time.perf_counter()

    gray = load_roi(
        image_path,
        preprocess_mode=preprocess_mode,
        enhancement_mode=enhancement_mode,
        mean_threshold=mean_threshold,
        p90_threshold=p90_threshold,
    )

    bars = build_topo_barcode(
        gray,
        threshold_mode=threshold_mode,
    )

    _ = barcode_to_vector(
        bars,
        gray=gray,
        threshold_mode=threshold_mode,
    )

    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0


def measure_full_match_time_ms(
    image_path: str,
    db_path: str,
    preprocess_mode: str,
    enhancement_mode: str,
    mean_threshold: float,
    p90_threshold: float,
    threshold_mode: str,
):
    """
    Замеряет полный match:
    построение query-дескриптора + сравнение с базой.
    """
    t0 = time.perf_counter()

    query_bars, results = match_against_database(
        query_image_path=image_path,
        db_path=db_path,
        preprocess_mode=preprocess_mode,
        enhancement_mode=enhancement_mode,
        mean_threshold=mean_threshold,
        p90_threshold=p90_threshold,
        threshold_mode=threshold_mode,
    )

    t1 = time.perf_counter()

    return results, (t1 - t0) * 1000.0


def evaluate_queries(
    query_dir: str,
    db_path: str,
    preprocess_mode: str,
    enhancement_mode: str,
    mean_threshold: float,
    p90_threshold: float,
    threshold_mode: str,
):
    rows = []
    query_paths = list_images(query_dir)

    if not query_paths:
        raise FileNotFoundError(f"Нет query-изображений в папке: {query_dir}")

    for query_path in query_paths:
        filename = os.path.basename(query_path)

        true_name = true_name_from_query(filename)
        true_class = class_from_name(true_name)
        variant = variant_from_query(filename)

        descriptor_ms = measure_descriptor_time_ms(
            image_path=query_path,
            preprocess_mode=preprocess_mode,
            enhancement_mode=enhancement_mode,
            mean_threshold=mean_threshold,
            p90_threshold=p90_threshold,
            threshold_mode=threshold_mode,
        )

        results, full_match_ms = measure_full_match_time_ms(
            image_path=query_path,
            db_path=db_path,
            preprocess_mode=preprocess_mode,
            enhancement_mode=enhancement_mode,
            mean_threshold=mean_threshold,
            p90_threshold=p90_threshold,
            threshold_mode=threshold_mode,
        )

        top1 = results[0]
        top3 = results[:3]

        top1_name = top1["name"]
        top1_class = class_from_name(top1_name)

        top3_names = [r["name"] for r in top3]
        top3_classes = [class_from_name(r["name"]) for r in top3]

        row = {
            "file": filename,
            "true_name": true_name,
            "true_class": true_class,
            "variant": variant,
            "top1_name": top1_name,
            "top1_class": top1_class,
            "top1_distance": top1["distance"],
            "top1_similarity": top1["similarity"],
            "top3_names": "|".join(top3_names),
            "top3_classes": "|".join(top3_classes),
            "top1_exact_ok": int(top1_name == true_name),
            "top1_class_ok": int(top1_class == true_class),
            "top3_exact_ok": int(true_name in top3_names),
            "top3_class_ok": int(true_class in top3_classes),
            "descriptor_ms": descriptor_ms,
            "full_match_ms": full_match_ms,
            "preprocess_mode": preprocess_mode,
            "enhancement_mode": enhancement_mode,
            "threshold_mode": threshold_mode,
        }

        rows.append(row)

    return rows


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize(rows, preprocess_mode, enhancement_mode, mean_threshold, p90_threshold, threshold_mode):
    n = len(rows)

    top1_exact = mean([r["top1_exact_ok"] for r in rows]) * 100
    top1_class = mean([r["top1_class_ok"] for r in rows]) * 100
    top3_exact = mean([r["top3_exact_ok"] for r in rows]) * 100
    top3_class = mean([r["top3_class_ok"] for r in rows]) * 100

    descriptor_time = mean([r["descriptor_ms"] for r in rows])
    full_match_time = mean([r["full_match_ms"] for r in rows])

    lines = []

    lines.append("=== РЕЖИМ ===")
    lines.append(f"preprocess_mode: {preprocess_mode}")
    lines.append(f"enhancement_mode: {enhancement_mode}")
    lines.append(f"mean_threshold: {float(mean_threshold)}")
    lines.append(f"p90_threshold: {float(p90_threshold)}")
    lines.append(f"threshold_mode: {threshold_mode}")
    lines.append("")

    lines.append("=== ОБЩИЕ ИТОГИ ===")
    lines.append(f"Количество query: {n}")
    lines.append(f"Top-1 exact accuracy: {top1_exact:.2f}%")
    lines.append(f"Top-1 class accuracy: {top1_class:.2f}%")
    lines.append(f"Top-3 exact accuracy: {top3_exact:.2f}%")
    lines.append(f"Top-3 class accuracy: {top3_class:.2f}%")
    lines.append(f"Среднее время дескриптора: {descriptor_time:.3f} ms")
    lines.append(f"Среднее полное время match: {full_match_time:.3f} ms")
    lines.append("")

    lines.append("=== ПО ТИПАМ ИСКАЖЕНИЙ ===")

    variants = sorted(set(r["variant"] for r in rows))

    for variant in variants:
        sub = [r for r in rows if r["variant"] == variant]

        lines.append(f"[{variant}]")
        lines.append(f"  N = {len(sub)}")
        lines.append(f"  Top-1 exact: {mean([r['top1_exact_ok'] for r in sub]) * 100:.2f}%")
        lines.append(f"  Top-1 class: {mean([r['top1_class_ok'] for r in sub]) * 100:.2f}%")
        lines.append(f"  Top-3 exact: {mean([r['top3_exact_ok'] for r in sub]) * 100:.2f}%")
        lines.append(f"  Top-3 class: {mean([r['top3_class_ok'] for r in sub]) * 100:.2f}%")
        lines.append(f"  Mean descriptor time: {mean([r['descriptor_ms'] for r in sub]):.3f} ms")
        lines.append(f"  Mean full match time: {mean([r['full_match_ms'] for r in sub]):.3f} ms")
        lines.append("")

    return "\n".join(lines)


def save_csv(rows, csv_path):
    ensure_dir(os.path.dirname(csv_path) or ".")

    fieldnames = [
        "file",
        "true_name",
        "true_class",
        "variant",
        "top1_name",
        "top1_class",
        "top1_distance",
        "top1_similarity",
        "top3_names",
        "top3_classes",
        "top1_exact_ok",
        "top1_class_ok",
        "top3_exact_ok",
        "top3_class_ok",
        "descriptor_ms",
        "full_match_ms",
        "preprocess_mode",
        "enhancement_mode",
        "threshold_mode",
    ]

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_text(text, path):
    ensure_dir(os.path.dirname(path) or ".")

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser(
        description="Robustness experiment for UAV topo-barcode prototype"
    )

    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--summary", required=True)

    parser.add_argument(
        "--preprocess-mode",
        default="none",
        choices=["none", "always", "auto"],
    )

    parser.add_argument(
        "--enhancement-mode",
        default="gamma_then_clahe",
        choices=["gamma", "clahe", "gamma_then_clahe"],
    )

    parser.add_argument("--mean-threshold", type=float, default=105.0)
    parser.add_argument("--p90-threshold", type=float, default=170.0)

    parser.add_argument(
        "--threshold-mode",
        default="dense",
        choices=["dense", "fixed", "quantile", "hybrid"],
        help="Режим выбора порогов: dense / fixed / quantile / hybrid"
    )

    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Не генерировать query заново, использовать уже существующие файлы"
    )

    args = parser.parse_args()

    if not args.no_generate:
        print("Шаг 1. Генерация искаженных query...")
        generated = generate_queries(args.ref_dir, args.query_dir)
        print(f"Сгенерировано файлов: {len(generated)}")
    else:
        print("Шаг 1. Генерация пропущена (--no-generate).")

    print("Шаг 2. Оценка устойчивости и времени...")
    rows = evaluate_queries(
        query_dir=args.query_dir,
        db_path=args.db,
        preprocess_mode=args.preprocess_mode,
        enhancement_mode=args.enhancement_mode,
        mean_threshold=args.mean_threshold,
        p90_threshold=args.p90_threshold,
        threshold_mode=args.threshold_mode,
    )

    print("Шаг 3. Сохранение результатов...")

    summary = summarize(
        rows=rows,
        preprocess_mode=args.preprocess_mode,
        enhancement_mode=args.enhancement_mode,
        mean_threshold=args.mean_threshold,
        p90_threshold=args.p90_threshold,
        threshold_mode=args.threshold_mode,
    )

    save_csv(rows, args.csv)
    save_text(summary, args.summary)

    print()
    print(summary)
    print()
    print(f"CSV сохранен: {args.csv}")
    print(f"Сводка сохранена: {args.summary}")


if __name__ == "__main__":
    main()