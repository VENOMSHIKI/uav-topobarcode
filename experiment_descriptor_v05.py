import argparse
import csv
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.preprocess import load_roi
from src.barcode import build_topo_barcode, barcode_to_vector

# Берем уже проверенные edge/shape-признаки из v0.4
from experiment_descriptor_v04 import edge_shape_features


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
    if path:
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


def safe_build_barcode(gray: np.ndarray, threshold_mode: str):
    try:
        return build_topo_barcode(gray, threshold_mode=threshold_mode)
    except TypeError:
        return build_topo_barcode(gray)


def safe_barcode_to_vector(bars, gray: np.ndarray, threshold_mode: str):
    try:
        return barcode_to_vector(bars, gray=gray, threshold_mode=threshold_mode)
    except TypeError:
        return barcode_to_vector(bars, gray=gray)


def parse_descriptor_mode(mode: str):
    """
    Режимы:
      barcode     -> только barcode-вектор
      edge100     -> barcode + edge/shape * 1.00
      texture025  -> barcode + edge/shape * 1.00 + texture/line * 0.25
      texture050  -> barcode + edge/shape * 1.00 + texture/line * 0.50
      texture075  -> barcode + edge/shape * 1.00 + texture/line * 0.75
      texture100  -> barcode + edge/shape * 1.00 + texture/line * 1.00
    """
    mode = (mode or "barcode").lower().strip()

    if mode == "barcode":
        return 0.0, 0.0

    if mode == "edge100":
        return 1.0, 0.0

    table = {
        "texture025": 0.25,
        "texture050": 0.50,
        "texture075": 0.75,
        "texture100": 1.00,
    }

    if mode in table:
        return 1.0, table[mode]

    raise ValueError(
        f"Неизвестный descriptor_mode: {mode}. "
        f"Доступно: barcode, edge100, texture025, texture050, texture075, texture100"
    )


def block_texture_features(gray: np.ndarray, grid_size: int = 4):
    """
    Локальная текстура по блокам.
    Идея: вода/рябь часто дает другую локальную дисперсию, чем крыши,
    улицы и городские кварталы.

    Делим ROI на grid_size x grid_size блоков.
    Для каждого блока считаем:
    - среднюю яркость;
    - стандартное отклонение;
    - плотность Canny-границ.
    """
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    h, w = gray.shape[:2]

    med = float(np.median(gray))
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    if upper <= lower:
        upper = min(255, lower + 1)

    edges = cv2.Canny(gray, lower, upper)

    features = []

    for gy in range(grid_size):
        y1 = int(gy * h / grid_size)
        y2 = int((gy + 1) * h / grid_size)

        for gx in range(grid_size):
            x1 = int(gx * w / grid_size)
            x2 = int((gx + 1) * w / grid_size)

            block = gray[y1:y2, x1:x2]
            edge_block = edges[y1:y2, x1:x2]

            area = max(1, block.shape[0] * block.shape[1])

            block_mean = float(np.mean(block)) / 255.0
            block_std = float(np.std(block)) / 128.0
            block_edge_density = float(np.count_nonzero(edge_block)) / float(area)

            features.extend([
                np.clip(block_mean, 0.0, 1.0),
                np.clip(block_std, 0.0, 1.0),
                np.clip(block_edge_density, 0.0, 1.0),
            ])

    return np.array(features, dtype=np.float32)


def simple_lbp_histogram(gray: np.ndarray, bins: int = 16):
    """
    Простая LBP-подобная текстура без сторонних библиотек.

    Для каждого пикселя сравниваем 8 соседей с центром.
    Получаем 8-битный код, затем собираем гистограмму.
    Чтобы вектор был компактным, используем 16 bins.
    """
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    padded = np.pad(gray, 1, mode="edge")
    center = padded[1:-1, 1:-1]

    code = np.zeros_like(center, dtype=np.uint8)

    neighbors = [
        padded[:-2, :-2],   # top-left
        padded[:-2, 1:-1],  # top
        padded[:-2, 2:],    # top-right
        padded[1:-1, 2:],   # right
        padded[2:, 2:],     # bottom-right
        padded[2:, 1:-1],   # bottom
        padded[2:, :-2],    # bottom-left
        padded[1:-1, :-2],  # left
    ]

    for idx, nb in enumerate(neighbors):
        code |= ((nb >= center).astype(np.uint8) << idx)

    # Уплотняем 256 кодов в 16 корзин
    compact_code = (code // max(1, 256 // bins)).astype(np.uint8)
    hist, _ = np.histogram(compact_code, bins=bins, range=(0, bins))

    hist = hist.astype(np.float32)
    hist = hist / max(1.0, float(hist.sum()))

    return hist.astype(np.float32)


def hough_line_features(gray: np.ndarray):
    """
    Признаки прямых линий.

    Городские кварталы, крыши и улицы часто дают жесткие прямые линии.
    Река чаще дает плавные границы и менее регулярную линейную структуру.
    """
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    h, w = gray.shape[:2]
    diag = float(np.sqrt(h * h + w * w))

    med = float(np.median(gray))
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    if upper <= lower:
        upper = min(255, lower + 1)

    edges = cv2.Canny(gray, lower, upper)

    min_line_length = max(8, int(0.18 * min(h, w)))
    max_line_gap = max(2, int(0.04 * min(h, w)))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=18,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        return np.zeros(11, dtype=np.float32)

    lengths = []
    angle_bins = np.zeros(4, dtype=np.float32)

    # angle bins:
    # 0 — почти горизонтальные
    # 1 — диагональ /
    # 2 — почти вертикальные
    # 3 — диагональ \
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line

        dx = float(x2 - x1)
        dy = float(y2 - y1)

        length = float(np.sqrt(dx * dx + dy * dy))
        if length <= 0:
            continue

        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle > 90:
            angle = 180 - angle

        lengths.append(length)

        if angle < 22.5:
            angle_bins[0] += length
        elif angle < 45.0:
            angle_bins[1] += length
        elif angle < 67.5:
            angle_bins[3] += length
        else:
            angle_bins[2] += length

    if not lengths:
        return np.zeros(11, dtype=np.float32)

    lengths = np.array(lengths, dtype=np.float32)

    line_count_norm = min(len(lengths) / 30.0, 1.0)
    total_len_norm = min(float(np.sum(lengths)) / (diag * 12.0), 1.0)
    mean_len_norm = min(float(np.mean(lengths)) / diag, 1.0)
    max_len_norm = min(float(np.max(lengths)) / diag, 1.0)
    std_len_norm = min(float(np.std(lengths)) / diag, 1.0)

    angle_sum = float(np.sum(angle_bins))
    if angle_sum > 0:
        angle_bins = angle_bins / angle_sum

    # Регулярность: если одна ориентация доминирует, значение выше.
    dominant_orientation = float(np.max(angle_bins))

    # Баланс направлений: у городской сетки часто несколько направлений,
    # у линейной улицы или крыши может доминировать одно.
    orientation_entropy = 0.0
    for v in angle_bins:
        if v > 0:
            orientation_entropy -= float(v) * float(np.log(v + 1e-9))
    orientation_entropy = orientation_entropy / np.log(4.0)

    features = [
        line_count_norm,
        total_len_norm,
        mean_len_norm,
        max_len_norm,
        std_len_norm,
        float(angle_bins[0]),
        float(angle_bins[1]),
        float(angle_bins[2]),
        float(angle_bins[3]),
        dominant_orientation,
        orientation_entropy,
    ]

    return np.array(features, dtype=np.float32)


def texture_line_features(gray: np.ndarray):
    """
    Итоговый texture/line-вектор.
    """
    block_vec = block_texture_features(gray, grid_size=4)
    lbp_vec = simple_lbp_histogram(gray, bins=16)
    line_vec = hough_line_features(gray)

    return np.concatenate([block_vec, lbp_vec, line_vec]).astype(np.float32)


def compute_descriptor(
    image_path: str,
    preprocess_mode: str,
    enhancement_mode: str,
    mean_threshold: float,
    p90_threshold: float,
    threshold_mode: str,
    descriptor_mode: str,
):
    gray = load_roi(
        image_path,
        preprocess_mode=preprocess_mode,
        enhancement_mode=enhancement_mode,
        mean_threshold=mean_threshold,
        p90_threshold=p90_threshold,
    )

    bars = safe_build_barcode(gray, threshold_mode=threshold_mode)
    barcode_vec = safe_barcode_to_vector(bars, gray=gray, threshold_mode=threshold_mode)
    barcode_vec = np.asarray(barcode_vec, dtype=np.float32)

    edge_weight, texture_weight = parse_descriptor_mode(descriptor_mode)

    vectors = [barcode_vec]

    if edge_weight > 0:
        edge_vec = edge_shape_features(gray) * float(edge_weight)
        vectors.append(edge_vec.astype(np.float32))

    if texture_weight > 0:
        texture_vec = texture_line_features(gray) * float(texture_weight)
        vectors.append(texture_vec.astype(np.float32))

    full_vec = np.concatenate(vectors).astype(np.float32)

    return gray, bars, full_vec


def build_records(
    ref_dir: str,
    preprocess_mode: str,
    enhancement_mode: str,
    mean_threshold: float,
    p90_threshold: float,
    threshold_mode: str,
    descriptor_mode: str,
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
            descriptor_mode=descriptor_mode,
        )

        records.append({
            "name": name,
            "class": class_from_name(name),
            "image_path": ref_path,
            "bars": bars,
            "vector": vector,
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
    descriptor_mode: str,
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
            descriptor_mode=descriptor_mode,
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
            "descriptor_mode": descriptor_mode,
            "vector_length": len(query_vector),
        })

    return rows


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def pct(values):
    return mean(values) * 100.0


def summarize_rows(rows, preprocess_mode, enhancement_mode, threshold_mode, descriptor_mode):
    n = len(rows)

    lines = []
    lines.append("=== РЕЖИМ ===")
    lines.append(f"preprocess_mode: {preprocess_mode}")
    lines.append(f"enhancement_mode: {enhancement_mode}")
    lines.append(f"threshold_mode: {threshold_mode}")
    lines.append(f"descriptor_mode: {descriptor_mode}")
    lines.append(f"vector_length: {rows[0]['vector_length'] if rows else 'NA'}")
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

    lines.append("=== ПО КЛАССАМ ===")

    by_class = defaultdict(list)
    for r in rows:
        by_class[r["true_class"]].append(r)

    for cls in sorted(by_class.keys()):
        sub = by_class[cls]

        lines.append(
            f"[{cls}] N={len(sub)} | "
            f"Top-1 exact={pct(r['top1_exact_ok'] for r in sub):.2f}% | "
            f"Top-1 class={pct(r['top1_class_ok'] for r in sub):.2f}% | "
            f"Top-3 exact={pct(r['top3_exact_ok'] for r in sub):.2f}% | "
            f"Top-3 class={pct(r['top3_class_ok'] for r in sub):.2f}%"
        )

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


def one_line_metrics(rows, descriptor_mode):
    return {
        "descriptor_mode": descriptor_mode,
        "query_count": len(rows),
        "vector_length": rows[0]["vector_length"] if rows else 0,
        "top1_exact": round(pct(r["top1_exact_ok"] for r in rows), 2),
        "top1_class": round(pct(r["top1_class_ok"] for r in rows), 2),
        "top3_exact": round(pct(r["top3_exact_ok"] for r in rows), 2),
        "top3_class": round(pct(r["top3_class_ok"] for r in rows), 2),
        "descriptor_ms": round(mean(r["descriptor_ms"] for r in rows), 3),
        "full_match_ms": round(mean(r["full_match_ms"] for r in rows), 3),
    }


def run_mode(args, descriptor_mode: str):
    print()
    print(f"===== DESCRIPTOR MODE: {descriptor_mode} =====")
    print("Шаг 1. Строим базу дескрипторов...")

    records = build_records(
        ref_dir=args.ref_dir,
        preprocess_mode=args.preprocess_mode,
        enhancement_mode=args.enhancement_mode,
        mean_threshold=args.mean_threshold,
        p90_threshold=args.p90_threshold,
        threshold_mode=args.threshold_mode,
        descriptor_mode=descriptor_mode,
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
        descriptor_mode=descriptor_mode,
    )

    csv_path = f"{args.out_prefix}_{descriptor_mode}_results.csv"
    summary_path = f"{args.out_prefix}_{descriptor_mode}_summary.txt"

    summary = summarize_rows(
        rows=rows,
        preprocess_mode=args.preprocess_mode,
        enhancement_mode=args.enhancement_mode,
        threshold_mode=args.threshold_mode,
        descriptor_mode=descriptor_mode,
    )

    save_csv(rows, csv_path)
    save_text(summary, summary_path)

    print(summary)
    print(f"CSV сохранен: {csv_path}")
    print(f"Сводка сохранена: {summary_path}")

    return one_line_metrics(rows, descriptor_mode)


def main():
    parser = argparse.ArgumentParser(
        description="v0.5 descriptor ablation: barcode + edge/shape + texture/line features"
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
        "--descriptor-modes",
        default="barcode,edge100,texture025,texture050,texture075,texture100",
        help="Через запятую: barcode,edge100,texture025,texture050,texture075,texture100"
    )

    args = parser.parse_args()

    modes = [m.strip() for m in args.descriptor_modes.split(",") if m.strip()]
    combined = []

    for mode in modes:
        combined.append(run_mode(args, mode))

    combined_csv = f"{args.out_prefix}_combined_summary.csv"
    ensure_dir(os.path.dirname(combined_csv) or ".")

    with open(combined_csv, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "descriptor_mode",
            "query_count",
            "vector_length",
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
    print("===== ОБЩЕЕ СРАВНЕНИЕ DESCRIPTOR V0.5 =====")

    for row in combined:
        print(
            f"{row['descriptor_mode']}: "
            f"vector={row['vector_length']}, "
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
