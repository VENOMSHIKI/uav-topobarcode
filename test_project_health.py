import csv
import os
import sys
from pathlib import Path

import numpy as np

from src.preprocess import load_roi
from experiment_descriptor_v05 import (
    list_images,
    safe_build_barcode,
    safe_barcode_to_vector,
    edge_shape_features,
    texture_line_features,
)


REF_DIR = r"data\reference_TheWorld50"
QUERY_DIR = r"data\query\TheWorld50_full_v05_clean_generated"
SUMMARY_CSV = r"outputs\logs\TheWorld50_full_v05_clean_combined_summary.csv"


def fail(message: str):
    print(f"[FAIL] {message}")
    sys.exit(1)


def ok(message: str):
    print(f"[OK] {message}")


def check_path_exists(path: str, label: str):
    if not os.path.exists(path):
        fail(f"{label} не найден: {path}")
    ok(f"{label} найден: {path}")


def check_image_counts():
    ref_images = list_images(REF_DIR)
    query_images = list_images(QUERY_DIR)

    if len(ref_images) != 80:
        fail(f"Ожидалось 80 эталонов, найдено: {len(ref_images)}")

    if len(query_images) != 480:
        fail(f"Ожидалось 480 query, найдено: {len(query_images)}")

    ok(f"Количество эталонов корректно: {len(ref_images)}")
    ok(f"Количество query корректно: {len(query_images)}")

    return ref_images, query_images


def build_texture100_vector(image_path: str):
    gray = load_roi(
        image_path,
        preprocess_mode="none",
        enhancement_mode="gamma_then_clahe",
        mean_threshold=105.0,
        p90_threshold=170.0,
    )

    bars = safe_build_barcode(gray, threshold_mode="quantile")
    barcode_vec = safe_barcode_to_vector(bars, gray=gray, threshold_mode="quantile")

    barcode_vec = np.asarray(barcode_vec, dtype=np.float32)
    edge_vec = edge_shape_features(gray).astype(np.float32)
    texture_vec = texture_line_features(gray).astype(np.float32)

    full_vec = np.concatenate([barcode_vec, edge_vec, texture_vec]).astype(np.float32)

    return bars, full_vec


def check_descriptor_vector(ref_images):
    image_path = ref_images[0]

    bars, vector = build_texture100_vector(image_path)

    if len(vector) != 152:
        fail(f"Ожидалась длина вектора 152, получено: {len(vector)}")

    if len(bars) <= 0:
        fail("Barcode bars не построились")

    if not np.all(np.isfinite(vector)):
        fail("Вектор содержит NaN или inf")

    ok(f"texture100 vector length = {len(vector)}")
    ok(f"barcode bars count = {len(bars)}")
    ok("Вектор не содержит NaN/inf")


def read_summary_csv():
    check_path_exists(SUMMARY_CSV, "Итоговая CSV-таблица v0.5 full clean")

    with open(SUMMARY_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        fail("CSV summary пустой")

    return rows


def find_row(rows, descriptor_mode: str):
    for row in rows:
        if row.get("descriptor_mode") == descriptor_mode:
            return row

    fail(f"В summary нет строки descriptor_mode={descriptor_mode}")


def to_float(row, key: str):
    try:
        return float(row[key])
    except Exception:
        fail(f"Не удалось прочитать поле {key} из строки: {row}")


def check_metrics(rows):
    texture100 = find_row(rows, "texture100")

    query_count = int(float(texture100["query_count"]))
    vector_length = int(float(texture100["vector_length"]))

    top1_exact = to_float(texture100, "top1_exact")
    top1_class = to_float(texture100, "top1_class")
    top3_exact = to_float(texture100, "top3_exact")
    top3_class = to_float(texture100, "top3_class")
    full_match_ms = to_float(texture100, "full_match_ms")

    if query_count != 480:
        fail(f"texture100 query_count должен быть 480, получено: {query_count}")

    if vector_length != 152:
        fail(f"texture100 vector_length должен быть 152, получено: {vector_length}")

    # Контрольные пороги чуть ниже фактических, чтобы тест не падал от небольшого разброса времени/окружения.
    if top1_exact < 93.0:
        fail(f"Top-1 exact ниже порога: {top1_exact}")

    if top1_class < 95.0:
        fail(f"Top-1 class ниже порога: {top1_class}")

    if top3_exact < 95.0:
        fail(f"Top-3 exact ниже порога: {top3_exact}")

    if top3_class < 98.0:
        fail(f"Top-3 class ниже порога: {top3_class}")

    if full_match_ms > 120.0:
        fail(f"full_match_ms слишком большой: {full_match_ms}")

    ok(f"texture100 query_count = {query_count}")
    ok(f"texture100 vector_length = {vector_length}")
    ok(f"Top-1 exact = {top1_exact}%")
    ok(f"Top-1 class = {top1_class}%")
    ok(f"Top-3 exact = {top3_exact}%")
    ok(f"Top-3 class = {top3_class}%")
    ok(f"full_match_ms = {full_match_ms} ms")


def main():
    print("=== UAV TOPOBARCODE PROJECT HEALTH TEST ===")

    check_path_exists(REF_DIR, "Папка эталонов")
    check_path_exists(QUERY_DIR, "Папка query")
    check_path_exists(SUMMARY_CSV, "CSV summary")

    ref_images, _ = check_image_counts()

    check_descriptor_vector(ref_images)

    rows = read_summary_csv()
    check_metrics(rows)

    print("")
    print("=== RESULT ===")
    print("Все контрольные проверки пройдены. Текущая рабочая версия: v0.5 texture100.")


if __name__ == "__main__":
    main()