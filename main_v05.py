import argparse
import json
import os
from pathlib import Path

import numpy as np

from experiment_descriptor_v05 import (
    build_records,
    compute_descriptor,
    match_query,
)


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def save_database(records, db_path: str, meta: dict):
    payload = {
        "meta": meta,
        "records": [],
    }

    for rec in records:
        payload["records"].append({
            "name": rec["name"],
            "class": rec["class"],
            "image_path": rec["image_path"],
            "vector": rec["vector"].tolist(),
            "bars_count": len(rec.get("bars", [])),
        })

    ensure_dir(os.path.dirname(db_path) or ".")

    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_database(db_path: str):
    with open(db_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    meta = payload.get("meta", {})
    records = []

    for rec in payload.get("records", []):
        records.append({
            "name": rec["name"],
            "class": rec["class"],
            "image_path": rec.get("image_path", ""),
            "vector": np.array(rec["vector"], dtype=np.float32),
        })

    return meta, records


def check_meta(meta: dict, args):
    expected = {
        "version": "v0.5",
        "descriptor_mode": args.descriptor_mode,
        "threshold_mode": args.threshold_mode,
        "preprocess_mode": args.preprocess_mode,
        "enhancement_mode": args.enhancement_mode,
    }

    mismatches = []

    for key, value in expected.items():
        if meta.get(key) != value:
            mismatches.append(f"{key}: db={meta.get(key)} vs query={value}")

    if mismatches:
        raise ValueError(
            "Несовпадение параметров базы и запроса:\n- "
            + "\n- ".join(mismatches)
            + "\nПострой базу заново с теми же параметрами."
        )


def cmd_build_db(args):
    records = build_records(
        ref_dir=args.ref_dir,
        preprocess_mode=args.preprocess_mode,
        enhancement_mode=args.enhancement_mode,
        mean_threshold=args.mean_threshold,
        p90_threshold=args.p90_threshold,
        threshold_mode=args.threshold_mode,
        descriptor_mode=args.descriptor_mode,
    )

    meta = {
        "version": "v0.5",
        "descriptor_mode": args.descriptor_mode,
        "threshold_mode": args.threshold_mode,
        "preprocess_mode": args.preprocess_mode,
        "enhancement_mode": args.enhancement_mode,
        "mean_threshold": args.mean_threshold,
        "p90_threshold": args.p90_threshold,
        "vector_length": int(len(records[0]["vector"])) if records else 0,
        "records_count": len(records),
    }

    save_database(records, args.db, meta)

    print(f"База v0.5 построена: {args.db}")
    print(f"Количество эталонов: {len(records)}")
    print(f"descriptor_mode: {args.descriptor_mode}")
    print(f"threshold_mode: {args.threshold_mode}")
    print(f"vector_length: {meta['vector_length']}")


def cmd_match(args):
    meta, records = load_database(args.db)
    check_meta(meta, args)

    _, bars, query_vector = compute_descriptor(
        image_path=args.image,
        preprocess_mode=args.preprocess_mode,
        enhancement_mode=args.enhancement_mode,
        mean_threshold=args.mean_threshold,
        p90_threshold=args.p90_threshold,
        threshold_mode=args.threshold_mode,
        descriptor_mode=args.descriptor_mode,
    )

    results = match_query(query_vector, records)

    print("=== V0.5 QUERY DESCRIPTOR ===")
    print(f"image: {args.image}")
    print(f"bars: {len(bars)}")
    print(f"vector_length: {len(query_vector)}")
    print("")

    print(f"=== TOP-{args.top_k} RESULTS ===")

    for i, item in enumerate(results[:args.top_k], start=1):
        print(
            f"{i}. {item['name']} | "
            f"class={item['class']} | "
            f"distance={item['distance']} | "
            f"similarity={item['similarity']} | "
            f"path={item.get('image_path', '')}"
        )


def add_common_args(parser):
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
        "--descriptor-mode",
        default="texture100",
        choices=["barcode", "edge100", "texture025", "texture050", "texture075", "texture100"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Main CLI for accepted UAV topological barcode v0.5 descriptor"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build_db_v05")
    build_parser.add_argument("--ref-dir", required=True)
    build_parser.add_argument("--db", required=True)
    add_common_args(build_parser)
    build_parser.set_defaults(func=cmd_build_db)

    match_parser = subparsers.add_parser("match_v05")
    match_parser.add_argument("--image", required=True)
    match_parser.add_argument("--db", required=True)
    match_parser.add_argument("--top-k", type=int, default=5)
    add_common_args(match_parser)
    match_parser.set_defaults(func=cmd_match)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()