import argparse
import csv
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def read_rows(csv_path: str):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def pct(part, total):
    return 100.0 * part / total if total else 0.0


def save_confusion_plot(labels, matrix, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(matrix)

    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=160)
    plt.close()


def analyze(csv_path: str, out_txt: str, out_png: str):
    rows = read_rows(csv_path)

    if not rows:
        raise ValueError(f"CSV пустой: {csv_path}")

    required = [
        "true_class",
        "top1_class",
        "true_name",
        "top1_name",
        "variant",
        "top1_exact_ok",
        "top1_class_ok",
        "top3_exact_ok",
        "top3_class_ok",
        "top1_distance",
        "top1_similarity",
    ]

    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise KeyError(f"В CSV нет колонок: {missing}")

    labels = sorted(set(r["true_class"] for r in rows) | set(r["top1_class"] for r in rows))

    y_true = [r["true_class"] for r in rows]
    y_pred = [r["top1_class"] for r in rows]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    save_confusion_plot(
        labels=labels,
        matrix=cm,
        out_path=out_png,
        title="Top-1 class confusion matrix",
    )

    total = len(rows)
    top1_exact = sum(int(r["top1_exact_ok"]) for r in rows)
    top1_class = sum(int(r["top1_class_ok"]) for r in rows)
    top3_exact = sum(int(r["top3_exact_ok"]) for r in rows)
    top3_class = sum(int(r["top3_class_ok"]) for r in rows)

    lines = []
    lines.append("=== CONFUSION ANALYSIS ===")
    lines.append(f"CSV: {csv_path}")
    lines.append(f"Количество query: {total}")
    lines.append("")
    lines.append("=== OVERALL ===")
    lines.append(f"Top-1 exact: {pct(top1_exact, total):.2f}%")
    lines.append(f"Top-1 class: {pct(top1_class, total):.2f}%")
    lines.append(f"Top-3 exact: {pct(top3_exact, total):.2f}%")
    lines.append(f"Top-3 class: {pct(top3_class, total):.2f}%")
    lines.append("")

    lines.append("=== CONFUSION MATRIX: TRUE x PRED ===")
    header = "true\\pred".ljust(14) + "".join(lbl[:12].rjust(14) for lbl in labels)
    lines.append(header)

    for i, true_label in enumerate(labels):
        line = true_label[:12].ljust(14)
        for j in range(len(labels)):
            line += str(cm[i, j]).rjust(14)
        lines.append(line)

    lines.append("")
    lines.append("=== PER CLASS ===")

    by_class = defaultdict(list)
    for r in rows:
        by_class[r["true_class"]].append(r)

    for cls in labels:
        sub = by_class[cls]
        n = len(sub)
        if n == 0:
            continue

        cls_top1_exact = sum(int(r["top1_exact_ok"]) for r in sub)
        cls_top1_class = sum(int(r["top1_class_ok"]) for r in sub)
        cls_top3_exact = sum(int(r["top3_exact_ok"]) for r in sub)
        cls_top3_class = sum(int(r["top3_class_ok"]) for r in sub)

        lines.append(
            f"{cls}: N={n}, "
            f"Top-1 exact={pct(cls_top1_exact, n):.2f}%, "
            f"Top-1 class={pct(cls_top1_class, n):.2f}%, "
            f"Top-3 exact={pct(cls_top3_exact, n):.2f}%, "
            f"Top-3 class={pct(cls_top3_class, n):.2f}%"
        )

    lines.append("")
    lines.append("=== CLASS CONFUSIONS TRUE -> PRED ===")

    class_confusions = Counter()
    for r in rows:
        if int(r["top1_class_ok"]) == 0:
            class_confusions[(r["true_class"], r["top1_class"])] += 1

    if class_confusions:
        for (true_cls, pred_cls), count in class_confusions.most_common():
            lines.append(f"{true_cls} -> {pred_cls}: {count}")
    else:
        lines.append("Нет ошибок по классам.")

    lines.append("")
    lines.append("=== EXACT CONFUSIONS TRUE_NAME -> TOP1_NAME ===")

    exact_confusions = Counter()
    for r in rows:
        if int(r["top1_exact_ok"]) == 0:
            exact_confusions[(r["true_name"], r["top1_name"])] += 1

    if exact_confusions:
        for (true_name, pred_name), count in exact_confusions.most_common(30):
            lines.append(f"{true_name} -> {pred_name}: {count}")
    else:
        lines.append("Нет ошибок по exact.")

    lines.append("")
    lines.append("=== ERRORS BY VARIANT ===")

    by_variant = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)

    for variant in sorted(by_variant.keys()):
        sub = by_variant[variant]
        n = len(sub)
        wrong_exact = sum(1 for r in sub if int(r["top1_exact_ok"]) == 0)
        wrong_class = sum(1 for r in sub if int(r["top1_class_ok"]) == 0)

        lines.append(
            f"{variant}: N={n}, "
            f"wrong_top1_exact={wrong_exact}, "
            f"wrong_top1_class={wrong_class}, "
            f"Top-1 exact={100.0 - pct(wrong_exact, n):.2f}%, "
            f"Top-1 class={100.0 - pct(wrong_class, n):.2f}%"
        )

    text = "\n".join(lines)

    ensure_dir(os.path.dirname(out_txt))
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)
    print()
    print(f"TXT сохранен: {out_txt}")
    print(f"PNG сохранен: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Analyze class confusion from experiment CSV")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-txt", required=True)
    parser.add_argument("--out-png", required=True)

    args = parser.parse_args()

    analyze(
        csv_path=args.csv,
        out_txt=args.out_txt,
        out_png=args.out_png,
    )


if __name__ == "__main__":
    main()