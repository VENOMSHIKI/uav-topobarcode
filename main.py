import argparse

from src.database import build_database, match_against_database


def add_common_processing_args(parser):
    """
    Общие параметры обработки ROI.

    Они должны быть одинаковыми при:
    1. построении базы;
    2. сопоставлении query с базой.
    """
    parser.add_argument(
        "--preprocess-mode",
        default="none",
        choices=["none", "always", "auto"],
        help="Режим предобработки ROI: none / always / auto"
    )

    parser.add_argument(
        "--enhancement-mode",
        default="gamma_then_clahe",
        choices=["gamma", "clahe", "gamma_then_clahe"],
        help="Способ усиления изображения для режимов always/auto"
    )

    parser.add_argument(
        "--mean-threshold",
        type=float,
        default=105.0,
        help="Порог средней яркости для auto-режима"
    )

    parser.add_argument(
        "--p90-threshold",
        type=float,
        default=170.0,
        help="Порог p90 яркости для auto-режима"
    )

    parser.add_argument(
        "--threshold-mode",
        default="dense",
        choices=["dense", "fixed", "quantile", "hybrid"],
        help=(
            "Режим выбора порогов для баркода: "
            "dense — старая плотная сетка; "
            "fixed — фиксированные 80/120/160; "
            "quantile — адаптивные процентильные пороги; "
            "hybrid — смешанный экспериментальный режим"
        )
    )


def cmd_build_db(args):
    """
    Команда построения базы эталонных ROI.
    """
    build_database(
        reference_folder=args.ref_dir,
        db_path=args.db,
        plots_folder=args.plots,
        preprocess_mode=args.preprocess_mode,
        enhancement_mode=args.enhancement_mode,
        mean_threshold=args.mean_threshold,
        p90_threshold=args.p90_threshold,
        threshold_mode=args.threshold_mode,
    )


def cmd_match(args):
    """
    Команда сопоставления одного query ROI с базой.
    """
    query_bars, results = match_against_database(
        query_image_path=args.image,
        db_path=args.db,
        preprocess_mode=args.preprocess_mode,
        enhancement_mode=args.enhancement_mode,
        mean_threshold=args.mean_threshold,
        p90_threshold=args.p90_threshold,
        threshold_mode=args.threshold_mode,
    )

    top_k = max(1, int(args.top_k))

    print("=== QUERY BARCODE ===")
    print(f"Количество bars: {len(query_bars)}")
    print()

    print(f"=== TOP-{top_k} RESULTS ===")

    for i, item in enumerate(results[:top_k], start=1):
        print(
            f"{i}. {item['name']} | "
            f"distance={item['distance']} | "
            f"similarity={item['similarity']} | "
            f"path={item.get('image_path', '')}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="UAV topo-barcode prototype: build database and match ROI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    # build_db
    build_db_parser = subparsers.add_parser(
        "build_db",
        help="Построить базу barcode-дескрипторов по эталонным ROI"
    )

    build_db_parser.add_argument(
        "--ref-dir",
        required=True,
        help="Папка с эталонными ROI"
    )

    build_db_parser.add_argument(
        "--db",
        required=True,
        help="Путь для сохранения db.json"
    )

    build_db_parser.add_argument(
        "--plots",
        default=None,
        help="Папка для сохранения картинок barcode"
    )

    add_common_processing_args(build_db_parser)
    build_db_parser.set_defaults(func=cmd_build_db)

    # match
    match_parser = subparsers.add_parser(
        "match",
        help="Сопоставить один query ROI с базой"
    )

    match_parser.add_argument(
        "--image",
        "--query",
        dest="image",
        required=True,
        help="Путь к query ROI"
    )

    match_parser.add_argument(
        "--db",
        required=True,
        help="Путь к db.json"
    )

    match_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Сколько ближайших результатов показать"
    )

    add_common_processing_args(match_parser)
    match_parser.set_defaults(func=cmd_match)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()