import argparse
import csv
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.preprocess import load_roi
from experiment_descriptor_v05 import (
    list_images,
    class_from_name,
    true_name_from_query,
    variant_from_query,
    safe_build_barcode,
    safe_barcode_to_vector,
    edge_shape_features,
    texture_line_features,
)


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def pct(values):
    return mean(values) * 100.0


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def transform_gray(gray: np.ndarray, transform_name: str) -> np.ndarray:
    """
    raw       — без изменений
    median3   — медианный фильтр 3x3
    median5   — медианный фильтр 5x5
    openclose — grayscale opening + closing 3x3
    closeopen — grayscale closing + opening 3x3
    """
    gray = normalize_gray(gray)
    transform_name = transform_name.lower().strip()

    if transform_name == "raw":
        return gray

    if transform_name == "median3":
        return cv2.medianBlur(gray, 3)

    if transform_name == "median5":
        return cv2.medianBlur(gray, 5)

    kernel = np.ones((3, 3), np.uint8)

    if transform_name == "openclose":
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    if transform_name == "closeopen":
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    raise ValueError(f"Неизвестный transform_name: {transform_name}")


def strategy_to_transforms(strategy: str):
    """
    single       -> только raw
    raw_median3  -> raw + median3
    raw_median5  -> raw + median5
    raw_morph    -> raw + openclose + closeopen
    robust_all   -> raw + median3 + median5 + openclose + closeopen
    """
    strategy = strategy.lower().strip()

    table = {
        "single": ["raw"],
        "raw_median3": ["raw", "median3"],
        "raw_median5": ["raw", "median5"],
        "raw_morph": ["raw", "openclose", "closeopen"],
        "robust_all": ["raw", "median3", "median5", "openclose", "closeopen"],
    }

    if strategy not in table:
        raise ValueError(
            f"Неизвестная strategy: {strategy}. "
            f"Доступно: {', '.join(table.keys())}"
        )

    return table[strategy]


def vector_from_gray(
    gray: np.ndarray,
    threshold_mode: str,
    descriptor_mode: str,
):
    """
    descriptor_mode:
      barcode    = только топологический barcode
      edge100    = barcode + edge/shape
      texture100 = barcode + edge/shape + texture/line
    """
    gray = normalize_gray(gray)
    descriptor_mode = descriptor_mode.lower().strip()

    bars = safe_build_barcode(gray, threshold_mode=threshold_mode)
    barcode_vec = safe_barcode_to_vector(bars, gray=gray, threshold_mode=threshold_mode)
    barcode_vec = np.asarray(barcode_vec, dtype=np.float32)

    vectors = [barcode_vec]

    if descriptor_mode in {"edge100", "texture100"}:
        vectors.append(edge_shape_features(gray).astype(np.float32))

    if descriptor_mode == "texture100":
        vectors.append(texture_line_features(gray).astype(np.float32))

    if descriptor_mode not in {"barcode", "edge100", "texture100"}:
        raise ValueError(
            f"Неизвестный descriptor_mode: {descriptor_mode}. "
            f"Доступно: barcode, edge100, texture100"
        )

    full_vec = np.concatenate(vectors).astype(np.float32)

    return bars, full_vec


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

        gray = load_roi(
            ref_path,
            preprocess_mode=preprocess_mode,
            enhancement_mode=enhancement_mode,
            mean_threshold=mean_threshold,
            p90_threshold=p90_threshold,
        )

        bars, vector = vector_from_gray(
            gray=gray,
            threshold_mode=threshold_mode,
            descriptor_mode=descriptor_mode,
        )

        records.append({
            "name": name,
            "class": class_from_name(name),
            "image_path": ref_path,
            "bars_count": len(bars),
            "vector": vector,
        })

    return records


def build_query_vectors(
    image_path: str,
    transforms,
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

    query_vectors = []

    for tr in transforms:
        tr_gray = transform_gray(gray, tr)

        bars, vector = vector_from_gray(
            gray=tr_gray,
            threshold_mode=threshold_mode,
            descriptor_mode=descriptor_mode,
        )

        query_vectors.append({
            "transform": tr,
            "bars_count": len(bars),
            "vector": vector,
        })

    return query_vectors


def match_query_ensemble(query_vectors, records):
    """
    Для каждого эталона считаем расстояние до каждого query-варианта.
    Берем минимальное расстояние.
    """
    results = []

    for rec in records:
        ref_vector = rec["vector"]

        best_distance = None
        best_transform = None

        for qv in query_vectors:
            query_vector = qv["vector"]

            if len(query_vector) != len(ref_vector):
                raise ValueError(
                    f"Несовпадение длины векторов: "
                    f"query={len(query_vector)}, reference={len(ref_vector)}, "
                    f"record={rec['name']}, transform={qv['transform']}"
                )

            distance = float(np.linalg.norm(query_vector - ref_vector))

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_transform = qv["transform"]

        similarity = 1.0 / (1.0 + best_distance)

        results.append({
            "name": rec["name"],
            "class": rec["class"],
            "image_path": rec["image_path"],
            "distance": round(best_distance, 6),
            "similarity": round(similarity, 6),
            "best_transform": best_transform,
        })

    results.sort(key=lambda x: x["distance"])
    return results


def evaluate_queries(
    query_dir: str,
    records,
    strategy: str,
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

    transforms = strategy_to_transforms(strategy)
    rows = []

    for query_path in query_paths:
        filename = os.path.basename(query_path)

        true_name = true_name_from_query(filename)
        true_class = class_from_name(true_name)
        variant = variant_from_query(filename)

        t0 = time.perf_counter()

        query_vectors = build_query_vectors(
            image_path=query_path,
            transforms=transforms,
            preprocess_mode=preprocess_mode,
            enhancement_mode=enhancement_mode,
            mean_threshold=mean_threshold,
            p90_threshold=p90_threshold,
            threshold_mode=threshold_mode,
            descriptor_mode=descriptor_mode,
        )

        t1 = time.perf_counter()

        results = match_query_ensemble(query_vectors, records)

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
            "top1_best_transform": top1["best_transform"],
            "top3_names": "|".join(top3_names),
            "top3_classes": "|".join(top3_classes),
            "top1_exact_ok": int(top1_name == true_name),
            "top1_class_ok": int(top1_class == true_class),
            "top3_exact_ok": int(true_name in top3_names),
            "top3_class_ok": int(true_class in top3_classes),
            "descriptor_ms": descriptor_ms,
            "full_match_ms": full_match_ms,
            "strategy": strategy,
            "transforms": "|".join(transforms),
            "preprocess_mode": preprocess_mode,
            "enhancement_mode": enhancement_mode,
            "threshold_mode": threshold_mode,
            "descriptor_mode": descriptor_mode,
            "vector_length": len(query_vectors[0]["vector"]),
        })

    return rows


def summarize_rows(rows, strategy, descriptor_mode, threshold_mode, preprocess_mode, enhancement_mode):
    n = len(rows)

    lines = []
    lines.append("=== РЕЖИМ ===")
    lines.append(f"strategy: {strategy}")
    lines.append(f"descriptor_mode: {descriptor_mode}")
    lines.append(f"threshold_mode: {threshold_mode}")
    lines.append(f"preprocess_mode: {preprocess_mode}")
    lines.append(f"enhancement_mode: {enhancement_mode}")
    lines.append(f"vector_length: {rows[0]['vector_length'] if rows else 'NA'}")
    lines.append(f"transforms: {rows[0]['transforms'] if rows else 'NA'}")
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

    lines.append("")
    lines.append("=== КАКОЙ QUERY-TRANSFORM ПОБЕДИЛ В TOP-1 ===")

    by_transform = defaultdict(int)
    for r in rows:
        by_transform[r["top1_best_transform"]] += 1

    for tr, count in sorted(by_transform.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"{tr}: {count}")

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


def one_line_metrics(rows, strategy):
    return {
        "strategy": strategy,
        "query_count": len(rows),
        "vector_length": rows[0]["vector_length"] if rows else 0,
        "transforms": rows[0]["transforms"] if rows else "",
        "top1_exact": round(pct(r["top1_exact_ok"] for r in rows), 2),
        "top1_class": round(pct(r["top1_class_ok"] for r in rows), 2),
        "top3_exact": round(pct(r["top3_exact_ok"] for r in rows), 2),
        "top3_class": round(pct(r["top3_class_ok"] for r in rows), 2),
        "descriptor_ms": round(mean(r["descriptor_ms"] for r in rows), 3),
        "full_match_ms": round(mean(r["full_match_ms"] for r in rows), 3),
    }


def run_strategy(args, strategy: str):
    print()
    print(f"===== QUERY ENSEMBLE STRATEGY: {strategy} =====")
    print("Шаг 1. Строим чистую базу эталонных дескрипторов...")

    records = build_records(
        ref_dir=args.ref_dir,
        preprocess_mode=args.preprocess_mode,
        enhancement_mode=args.enhancement_mode,
        mean_threshold=args.mean_threshold,
        p90_threshold=args.p90_threshold,
        threshold_mode=args.threshold_mode,
        descriptor_mode=args.descriptor_mode,
    )

    print(f"Количество эталонов: {len(records)}")
    print("Шаг 2. Оцениваем query ensemble...")

    rows = evaluate_queries(
        query_dir=args.query_dir,
        records=records,
        strategy=strategy,
        preprocess_mode=args.preprocess_mode,
        enhancement_mode=args.enhancement_mode,
        mean_threshold=args.mean_threshold,
        p90_threshold=args.p90_threshold,
        threshold_mode=args.threshold_mode,
        descriptor_mode=args.descriptor_mode,
    )

    csv_path = f"{args.out_prefix}_{strategy}_results.csv"
    summary_path = f"{args.out_prefix}_{strategy}_summary.txt"

    summary = summarize_rows(
        rows=rows,
        strategy=strategy,
        descriptor_mode=args.descriptor_mode,
        threshold_mode=args.threshold_mode,
        preprocess_mode=args.preprocess_mode,
        enhancement_mode=args.enhancement_mode,
    )

    save_csv(rows, csv_path)
    save_text(summary, summary_path)

    print(summary)
    print(f"CSV сохранен: {csv_path}")
    print(f"Сводка сохранена: {summary_path}")

    return one_line_metrics(rows, strategy)


def main():
    parser = argparse.ArgumentParser(
        description="v0.6 query ensemble: robust matching for noisy query"
    )

    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--out-prefix", required=True)

    parser.add_argument("--descriptor-mode", default="texture100", choices=["barcode", "edge100", "texture100"])
    parser.add_argument("--threshold-mode", default="quantile", choices=["dense", "fixed", "quantile", "hybrid"])

    parser.add_argument("--preprocess-mode", default="none")
    parser.add_argument("--enhancement-mode", default="gamma_then_clahe")
    parser.add_argument("--mean-threshold", type=float, default=105.0)
    parser.add_argument("--p90-threshold", type=float, default=170.0)

    parser.add_argument(
        "--strategies",
        default="single,raw_median3,raw_median5,raw_morph,robust_all",
        help="Через запятую: single,raw_median3,raw_median5,raw_morph,robust_all",
    )

    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    combined = []

    for strategy in strategies:
        combined.append(run_strategy(args, strategy))

    combined_csv = f"{args.out_prefix}_combined_summary.csv"
    ensure_dir(os.path.dirname(combined_csv) or ".")

    with open(combined_csv, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "strategy",
            "query_count",
            "vector_length",
            "transforms",
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
    print("===== ОБЩЕЕ СРАВНЕНИЕ QUERY ENSEMBLE V0.6 =====")

    for row in combined:
        print(
            f"{row['strategy']}: "
            f"vector={row['vector_length']}, "
            f"Top-1 exact={row['top1_exact']}%, "
            f"Top-1 class={row['top1_class']}%, "
            f"Top-3 exact={row['top3_exact']}%, "
            f"Top-3 class={row['top3_class']}%, "
            f"full_match={row['full_match_ms']} ms, "
            f"transforms={row['transforms']}"
        )

    print()
    print(f"Общая таблица сохранена: {combined_csv}")


if __name__ == "__main__":
    main()