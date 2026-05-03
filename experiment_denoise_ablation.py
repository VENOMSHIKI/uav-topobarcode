import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np

from src.preprocess import load_roi
from src.barcode import build_topo_barcode, barcode_to_vector


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: str):
    files = []

    if not os.path.isdir(folder):
        return files

    for root, _, names in os.walk(folder):
        for name in sorted(names):
            path = os.path.join(root, name)
            ext = os.path.splitext(name.lower())[1]

            if os.path.isfile(path) and ext in IMAGE_EXTENSIONS:
                files.append(path)

    return sorted(files)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def true_name_from_query(filename: str):
    stem = Path(filename).stem

    if "__" in stem:
        return stem.split("__")[0]

    return stem


def variant_from_query(filename: str):
    stem = Path(filename).stem

    if "__" in stem:
        return stem.split("__", 1)[1]

    return "orig"


def class_from_name(name: str):
    """
    bridge_08 -> bridge
    city_block_01 -> city_block
    """
    base = Path(name).stem
    parts = base.split("_")

    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])

    return parts[0]


def resize_keep_size(img, scale_factor: float):
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
    Генерирует query-изображения из эталонов.

    На каждый reference ROI:
    - orig
    - bright
    - dark
    - gauss
    - sp
    - scale
    """
    ensure_dir(query_dir)

    for old in list_images(query_dir):
        os.remove(old)

    ref_paths = list_images(ref_dir)

    if not ref_paths:
        raise FileNotFoundError(f"Нет изображений в ref-dir: {ref_dir}")

    generated = []

    np.random.seed(42)

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


def apply_denoise(gray: np.ndarray, denoise_mode: str):
    """
    Фильтрация перед построением баркода.

    none      — ничего не делаем
    median3   — медианный фильтр 3x3
    gaussian3 — гауссов фильтр 3x3
    """
    denoise_mode = (denoise_mode or "none").lower().strip()

    if denoise_mode == "none":
        return gray

    if denoise_mode == "median3":
        return cv2.medianBlur(gray, 3)

    if denoise_mode == "median5":
        return cv2.medianBlur(gray, 5)

    if denoise_mode == "gaussian3":
        return cv2.GaussianBlur(gray, (3, 3), 0)

    if denoise_mode == "gaussian5":
        return cv2.GaussianBlur(gray, (5, 5), 0)

    raise ValueError(
        f"Неизвестный denoise_mode: {denoise_mode}. "
        f"Доступно: none, median3, median5, gaussian3, gaussian5"
    )


def load_gray_for_experiment(
    image_path: str,
    preprocess_mode: str,
    enhancement_mode: str,
    mean_threshold: float,
    p90_threshold: float,
    denoise_mode: str,
):
    """
    Берем текущий load_roi из проекта, а затем добавляем denoise-фильтр.
    Так мы не ломаем старый рабочий код.
    """
    gray = load_roi(
        image_path,
        preprocess_mode=preprocess_mode,
        enhancement_mode=enhancement_mode,
        mean_threshold=mean_threshold,
        p90_threshold=p90_threshold,
    )

    gray = apply_denoise(gray, denoise_mode=denoise_mode)

    return gray


def compute_descriptor(
    image_path: str,
    preprocess_mode: str,
    enhancement_mode: str,
    mean_threshold: float,
    p90_threshold: float,
    threshold_mode: str,
    denoise_mode: str,
):
    gray = load_gray_for_experiment(
        image_path=image_path,
        preprocess_mode=preprocess_mode,
        enhancement_mode=enhancement_mode,
        mean_threshold=mean_threshold,
        p90_threshold=p90_threshold,
        denoise_mode=denoise_mode,
    )

    bars = build_topo_barcode(
        gray,
        threshold_mode=threshold_mode,
    )

    vector = barcode_to_vector(
        bars,
        gray=gray,
        threshold_mode=threshold_mode,
    )

    return gray, bars, vector


def build_records(
    ref_dir: str,
    preprocess_mode: str,
    enhancement_mode: str,
    mean_threshold: float,
    p90_threshold: float,
    threshold_mode: str,
    denoise_mode: str,
):
    ref_paths = list_images(ref_dir)

    if not ref_paths:
        raise FileNotFoundError(f"Нет изображений в ref-dir: {ref_dir}")

    records = []

    for ref_path in ref_paths:
        name = Path(ref_path).stem

        _, bars, vector = compute_descriptor(
            image_path=ref_path,
            preprocess_mode=preprocess_mode,
            enhancement_mode=enhancement_mode,
            mean_threshold=mean_threshold,
            p90_threshold=p90_threshold,
            threshold_mode=threshold_mode,
            denoise_mode=denoise_mode,
        )

        records.append({
            "name": name,
            "class": class_from_name(name),
            "image_path": ref_path,
            "bars": bars,
            "vector": vector.astype(np.float32),
        })

    return records


def match_query(query_vector: np.ndarray, records):
    results = []

    for rec in records:
        ref_vector = rec["vector"]

        if len(query_vector) != len(ref_vector):
            raise ValueError(
                f"Несовпадение длины векторов: "
                f"query={len(query_vector)}, reference={len(ref_vector)}, record={rec['name']}"
            )

        distance = float(np.linalg.norm(query_vector - ref_vector))
        similarity = 1.0 / (1.0 + distance)

        results.append({
            "name": rec["name"],
            "class": rec["class"],
            "image_path": rec["image_path"],
            "distance": round(distance, 6),
            "similarity": round(similarity, 6),
        })

    results.sort(key=lambda x: x["distance"])
    return results


def evaluate_queries(
    query_dir: str,
    records,
    preprocess_mode: str,
    enhancement_mode: str,
    mean_threshold: float,
    p90_threshold: float,
    threshold_mode: str,
    denoise_mode: str,
):
    query_paths = list_images(query_dir)

    if not query_paths:
        raise FileNotFoundError(f"Нет query-изображений в папке: {query_dir}")

    rows = []

    for query_path in query_paths:
        filename = os.path.basename(query_path)

        true_name = true_name_from_query(filename)
        true_class = class_from_name(true_name)
        variant = variant_from_query(filename)

        t0 = time.perf_counter()

        _, query_bars, query_vector = compute_descriptor(
            image_path=query_path,
            preprocess_mode=preprocess_mode,
            enhancement_mode=enhancement_mode,
            mean_threshold=mean_threshold,
            p90_threshold=p90_threshold,
            threshold_mode=threshold_mode,
            denoise_mode=denoise_mode,
        )

        t1 = time.perf_counter()

        results = match_query(query_vector, records)

        t2 = time.perf_counter()

        descriptor_ms = (t1 - t0) * 1000.0
        full_match_ms = (t2 - t0) * 1000.0

        top1 = results[0]
        top3 = results[:3]

        top1_name = top1["name"]
        top1_class = top1["class"]

        top3_names = [r["name"] for r in top3]
        top3_classes = [r["class"] for r in top3]

        rows.append({
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
            "denoise_mode": denoise_mode,
        })

    return rows


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def pct(values):
    return mean(values) * 100.0


def summarize_rows(rows, preprocess_mode, enhancement_mode, threshold_mode, denoise_mode):
    n = len(rows)

    lines = []
    lines.append("=== РЕЖИМ ===")
    lines.append(f"preprocess_mode: {preprocess_mode}")
    lines.append(f"enhancement_mode: {enhancement_mode}")
    lines.append(f"threshold_mode: {threshold_mode}")
    lines.append(f"denoise_mode: {denoise_mode}")
    lines.append("")

    lines.append("=== ОБЩИЕ ИТОГИ ===")
    lines.append(f"Количество query: {n}")
    lines.append(f"Top-1 exact accuracy: {pct(r['top1_exact_ok'] for r in rows):.2f}%")
    lines.append(f"Top-1 class accuracy: {pct(r['top1_class_ok'] for r in rows):.2f}%")
    lines.append(f"Top-3 exact accuracy: {pct(r['top3_exact_ok'] for r in rows):.2f}%")
    lines.append(f"Top-3 class accuracy: {pct(r['top3_class_ok'] for r in rows):.2f}%")
    lines.append(f"Среднее время дескриптора: {mean(r['descriptor_ms'] for r in rows):.3f} ms")
    lines.append(f"Среднее полное время match: {mean(r['full_match_ms'] for r in rows):.3f} ms")
    lines.append("")

    lines.append("=== ПО ТИПАМ ИСКАЖЕНИЙ ===")

    for variant in sorted(set(r["variant"] for r in rows)):
        sub = [r for r in rows if r["variant"] == variant]

        lines.append(f"[{variant}]")
        lines.append(f"  N = {len(sub)}")
        lines.append(f"  Top-1 exact: {pct(r['top1_exact_ok'] for r in sub):.2f}%")
        lines.append(f"  Top-1 class: {pct(r['top1_class_ok'] for r in sub):.2f}%")
        lines.append(f"  Top-3 exact: {pct(r['top3_exact_ok'] for r in sub):.2f}%")
        lines.append(f"  Top-3 class: {pct(r['top3_class_ok'] for r in sub):.2f}%")
        lines.append(f"  Mean descriptor time: {mean(r['descriptor_ms'] for r in sub):.3f} ms")
        lines.append(f"  Mean full match time: {mean(r['full_match_ms'] for r in sub):.3f} ms")
        lines.append("")

    return "\n".join(lines)


def save_csv(rows, csv_path):
    ensure_dir(os.path.dirname(csv_path) or ".")

    fieldnames = list(rows[0].keys()) if rows else []

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_text(text, path):
    ensure_dir(os.path.dirname(path) or ".")

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def one_line_metrics(rows, denoise_mode):
    return {
        "denoise_mode": denoise_mode,
        "query_count": len(rows),
        "top1_exact": round(pct(r["top1_exact_ok"] for r in rows), 2),
        "top1_class": round(pct(r["top1_class_ok"] for r in rows), 2),
        "top3_exact": round(pct(r["top3_exact_ok"] for r in rows), 2),
        "top3_class": round(pct(r["top3_class_ok"] for r in rows), 2),
        "descriptor_ms": round(mean(r["descriptor_ms"] for r in rows), 3),
        "full_match_ms": round(mean(r["full_match_ms"] for r in rows), 3),
    }


def run_mode(args, denoise_mode: str):
    print()
    print(f"===== DENOISE MODE: {denoise_mode} =====")
    print("Шаг 1. Строим базу дескрипторов...")

    records = build_records(
        ref_dir=args.ref_dir,
        preprocess_mode=args.preprocess_mode,
        enhancement_mode=args.enhancement_mode,
        mean_threshold=args.mean_threshold,
        p90_threshold=args.p90_threshold,
        threshold_mode=args.threshold_mode,
        denoise_mode=denoise_mode,
    )

    print(f"Количество эталонов: {len(records)}")
    print("Шаг 2. Оцениваем query...")

    rows = evaluate_queries(
        query_dir=args.query_dir,
        records=records,
        preprocess_mode=args.preprocess_mode,
        enhancement_mode=args.enhancement_mode,
        mean_threshold=args.mean_threshold,
        p90_threshold=args.p90_threshold,
        threshold_mode=args.threshold_mode,
        denoise_mode=denoise_mode,
    )

    csv_path = f"{args.out_prefix}_{denoise_mode}_results.csv"
    summary_path = f"{args.out_prefix}_{denoise_mode}_summary.txt"

    summary = summarize_rows(
        rows=rows,
        preprocess_mode=args.preprocess_mode,
        enhancement_mode=args.enhancement_mode,
        threshold_mode=args.threshold_mode,
        denoise_mode=denoise_mode,
    )

    save_csv(rows, csv_path)
    save_text(summary, summary_path)

    print(summary)
    print(f"CSV сохранен: {csv_path}")
    print(f"Сводка сохранена: {summary_path}")

    return one_line_metrics(rows, denoise_mode)


def main():
    parser = argparse.ArgumentParser(
        description="Denoise ablation for UAV topo-barcode experiment"
    )

    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--out-prefix", required=True)

    parser.add_argument("--preprocess-mode", default="none")
    parser.add_argument("--enhancement-mode", default="gamma_then_clahe")
    parser.add_argument("--mean-threshold", type=float, default=105.0)
    parser.add_argument("--p90-threshold", type=float, default=170.0)

    parser.add_argument(
        "--threshold-mode",
        default="quantile",
        choices=["dense", "fixed", "quantile", "hybrid"],
    )

    parser.add_argument(
        "--denoise-modes",
        default="none,median3,gaussian3",
        help="Через запятую: none,median3,median5,gaussian3,gaussian5"
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Сгенерировать query заново перед экспериментом"
    )

    args = parser.parse_args()

    if args.generate:
        print("Генерируем query...")
        generated = generate_queries(args.ref_dir, args.query_dir)
        print(f"Сгенерировано файлов: {len(generated)}")
    else:
        print("Генерация query пропущена. Используются существующие query.")

    modes = [m.strip() for m in args.denoise_modes.split(",") if m.strip()]
    combined = []

    for mode in modes:
        combined.append(run_mode(args, mode))

    combined_csv = f"{args.out_prefix}_combined_summary.csv"
    ensure_dir(os.path.dirname(combined_csv) or ".")

    with open(combined_csv, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "denoise_mode",
            "query_count",
            "top1_exact",
            "top1_class",
            "top3_exact",
            "top3_class",
            "descriptor_ms",
            "full_match_ms",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)

    print()
    print("===== ОБЩЕЕ СРАВНЕНИЕ DENOISE =====")
    for row in combined:
        print(
            f"{row['denoise_mode']}: "
            f"Top-1 exact={row['top1_exact']}%, "
            f"Top-1 class={row['top1_class']}%, "
            f"Top-3 exact={row['top3_exact']}%, "
            f"Top-3 class={row['top3_class']}%, "
            f"full_match={row['full_match_ms']} ms"
        )

    print()
    print(f"Общая таблица сохранена: {combined_csv}")


if __name__ == "__main__":
    main()