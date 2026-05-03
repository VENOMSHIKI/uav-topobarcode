import argparse
import csv
from collections import Counter, defaultdict


def get_first_existing(row, keys, default=""):
    """
    Берем первое найденное поле из списка возможных названий.
    Нужно, потому что в разных экспериментах CSV имеет разные имена колонок.
    """
    for key in keys:
        if key in row:
            return row[key]
    return default


def to_int_safe(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def to_float_safe(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def read_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def normalize_row(row):
    """
    Приводим строку из CSV к единому формату.
    Старый CSV и v0.3 CSV имеют разные названия колонок.
    """

    file_name = get_first_existing(row, ["file", "query_file", "filename", "image"])
    true_name = get_first_existing(row, ["true", "true_name", "expected", "expected_name"])
    pred_name = get_first_existing(row, ["top1", "pred", "pred_name", "top1_name"])

    variant = get_first_existing(row, ["variant", "distortion", "type"], default="unknown")

    # В старом CSV могло быть top1_exact_ok, в новом correct_top1
    top1_ok_raw = get_first_existing(row, ["top1_exact_ok", "correct_top1", "top1_ok"], default="")

    # В старом CSV могло быть top3_exact_ok, в новом correct_top3
    top3_ok_raw = get_first_existing(row, ["top3_exact_ok", "correct_top3", "top3_ok"], default="")

    top3_raw = get_first_existing(row, ["top3", "top3_names"], default="")

    distance = get_first_existing(row, ["dist", "distance", "top1_distance"], default="")
    similarity = get_first_existing(row, ["sim", "similarity", "top1_similarity"], default="")

    descriptor_ms = get_first_existing(row, ["descriptor_ms", "desc_ms"], default="")
    match_ms = get_first_existing(row, ["match_ms", "full_match_ms"], default="")

    # Если флага correct_top1 нет, вычислим его сами
    if top1_ok_raw == "":
        top1_ok = int(pred_name == true_name)
    else:
        top1_ok = to_int_safe(top1_ok_raw)

    # Если флага correct_top3 нет, вычислим сами по строке top3
    if top3_ok_raw == "":
        if top3_raw:
            top3_items = [x.strip() for x in top3_raw.replace(",", "|").split("|")]
            top3_ok = int(true_name in top3_items)
        else:
            top3_ok = 0
    else:
        top3_ok = to_int_safe(top3_ok_raw)

    return {
        "file": file_name,
        "true": true_name,
        "pred": pred_name,
        "variant": variant,
        "top3": top3_raw,
        "top1_ok": top1_ok,
        "top3_ok": top3_ok,
        "distance": to_float_safe(distance, default=0.0),
        "similarity": to_float_safe(similarity, default=0.0),
        "descriptor_ms": to_float_safe(descriptor_ms, default=0.0),
        "match_ms": to_float_safe(match_ms, default=0.0),
    }


def print_errors(rows):
    wrong_top1 = [r for r in rows if r["top1_ok"] == 0]
    wrong_top3 = [r for r in rows if r["top3_ok"] == 0]

    print("=== ОШИБКИ TOP-1 EXACT ===")
    if not wrong_top1:
        print("Нет ошибок.")
    else:
        for r in wrong_top1:
            print(
                f"{r['file']} | true={r['true']} | pred={r['pred']} | "
                f"variant={r['variant']} | dist={r['distance']:.6f} | sim={r['similarity']:.6f}"
            )

    print()
    print("=== ОШИБКИ TOP-3 EXACT ===")
    if not wrong_top3:
        print("Нет ошибок.")
    else:
        for r in wrong_top3:
            print(
                f"{r['file']} | true={r['true']} | top3={r['top3']} | "
                f"variant={r['variant']}"
            )


def print_summary_by_variant(rows):
    by_variant = defaultdict(list)

    for r in rows:
        by_variant[r["variant"]].append(r)

    print()
    print("=== СВОДКА ПО ИСКАЖЕНИЯМ ===")

    for variant in sorted(by_variant.keys()):
        items = by_variant[variant]
        n = len(items)

        wrong_top1 = sum(1 for r in items if r["top1_ok"] == 0)
        wrong_top3 = sum(1 for r in items if r["top3_ok"] == 0)

        top1_acc = (n - wrong_top1) / n * 100 if n else 0
        top3_acc = (n - wrong_top3) / n * 100 if n else 0

        mean_desc = sum(r["descriptor_ms"] for r in items) / n if n else 0
        mean_match = sum(r["match_ms"] for r in items) / n if n else 0

        print(
            f"{variant}: N={n}, "
            f"Top-1={top1_acc:.2f}%, wrong_top1={wrong_top1}, "
            f"Top-3={top3_acc:.2f}%, wrong_top3={wrong_top3}, "
            f"desc={mean_desc:.3f} ms, match={mean_match:.3f} ms"
        )


def print_confusions(rows):
    wrong_top1 = [r for r in rows if r["top1_ok"] == 0]

    print()
    print("=== ЧАСТЫЕ ПУТАНИЦЫ TRUE -> PRED ===")

    if not wrong_top1:
        print("Нет путаниц.")
        return

    pairs = Counter((r["true"], r["pred"]) for r in wrong_top1)

    for (true_name, pred_name), count in pairs.most_common():
        print(f"{true_name} -> {pred_name}: {count}")


def print_general(rows):
    n = len(rows)

    if n == 0:
        print("CSV пустой.")
        return

    top1 = sum(r["top1_ok"] for r in rows) / n * 100
    top3 = sum(r["top3_ok"] for r in rows) / n * 100

    mean_desc = sum(r["descriptor_ms"] for r in rows) / n
    mean_match = sum(r["match_ms"] for r in rows) / n

    print("=== ОБЩИЕ ИТОГИ ПО CSV ===")
    print(f"Количество query: {n}")
    print(f"Top-1 exact accuracy: {top1:.2f}%")
    print(f"Top-3 exact accuracy: {top3:.2f}%")
    print(f"Среднее время дескриптора: {mean_desc:.3f} ms")
    print(f"Среднее время match: {mean_match:.3f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    raw_rows, fieldnames = read_rows(args.csv)
    rows = [normalize_row(r) for r in raw_rows]

    print("=== ПРОВЕРКА CSV ===")
    print(f"Файл: {args.csv}")
    print(f"Колонки: {', '.join(fieldnames)}")
    print()

    print_general(rows)
    print()
    print_errors(rows)
    print_summary_by_variant(rows)
    print_confusions(rows)


if __name__ == "__main__":
    main()