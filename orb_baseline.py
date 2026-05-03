import argparse
import csv
import os
import statistics
import time

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


def load_gray(image_path: str, size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать: {image_path}")
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def name_to_class(name: str) -> str:
    parts = name.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return name


def parse_query_name(file_path: str):
    stem = os.path.splitext(os.path.basename(file_path))[0]
    true_name, variant = stem.split("__", 1)
    return true_name, variant


def build_orb_database(reference_dir: str):
    orb = cv2.ORB_create(nfeatures=300)
    db = []

    for path in list_images(reference_dir):
        gray = load_gray(path)
        kp, des = orb.detectAndCompute(gray, None)

        db.append({
            "name": os.path.splitext(os.path.basename(path))[0],
            "path": path,
            "kp_count": 0 if kp is None else len(kp),
            "des": des
        })

    return db


def match_orb(query_path: str, db):
    orb = cv2.ORB_create(nfeatures=300)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    gray = load_gray(query_path)
    kp_q, des_q = orb.detectAndCompute(gray, None)

    results = []
    for rec in db:
        des_r = rec["des"]

        if des_q is None or des_r is None or len(des_q) == 0 or len(des_r) == 0:
            score = 1e9
            good_matches = 0
        else:
            matches = bf.match(des_q, des_r)
            matches = sorted(matches, key=lambda m: m.distance)

            top_matches = matches[:30]
            if top_matches:
                score = float(np.mean([m.distance for m in top_matches]))
                good_matches = len(top_matches)
            else:
                score = 1e9
                good_matches = 0

        results.append({
            "name": rec["name"],
            "score": score,
            "good_matches": good_matches
        })

    results.sort(key=lambda x: (x["score"], -x["good_matches"]))
    return results


def main():
    parser = argparse.ArgumentParser(description="ORB baseline")
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    db = build_orb_database(args.ref_dir)
    query_paths = list_images(args.query_dir)

    rows = []
    total_times = []

    for query_path in query_paths:
        true_name, variant = parse_query_name(query_path)
        true_class = name_to_class(true_name)

        t0 = time.perf_counter()
        results = match_orb(query_path, db)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_times.append(elapsed_ms)

        top1_name = results[0]["name"]
        top1_class = name_to_class(top1_name)
        top3_names = [r["name"] for r in results[:3]]
        top3_classes = [name_to_class(x) for x in top3_names]

        rows.append({
            "query_file": os.path.basename(query_path),
            "true_name": true_name,
            "true_class": true_class,
            "variant": variant,
            "top1_name": top1_name,
            "top1_class": top1_class,
            "top1_exact_ok": int(top1_name == true_name),
            "top1_class_ok": int(top1_class == true_class),
            "top3_exact_ok": int(true_name in top3_names),
            "top3_class_ok": int(true_class in top3_classes),
            "best_score": results[0]["score"],
            "good_matches": results[0]["good_matches"],
            "full_match_ms": round(elapsed_ms, 3),
        })

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def acc(key):
        return 100.0 * sum(r[key] for r in rows) / len(rows)

    print("=== ORB BASELINE ===")
    print(f"N = {len(rows)}")
    print(f"Top-1 exact: {acc('top1_exact_ok'):.2f}%")
    print(f"Top-1 class: {acc('top1_class_ok'):.2f}%")
    print(f"Top-3 exact: {acc('top3_exact_ok'):.2f}%")
    print(f"Top-3 class: {acc('top3_class_ok'):.2f}%")
    print(f"Mean full match time: {statistics.mean(total_times):.3f} ms")
    print(f"CSV saved to: {args.csv}")


if __name__ == "__main__":
    main()