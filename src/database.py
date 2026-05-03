import json
import os
import numpy as np

from src.preprocess import load_roi
from src.barcode import build_topo_barcode, barcode_to_vector, plot_barcode


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: str):
    """
    Возвращает список изображений из папки и всех вложенных подпапок.

    Это нужно для смешанной базы:
    data/reference_TheWorld50/
        bridge/
        river/
        roof/
        street/
        city_block/
    """
    files = []

    for root, dirs, names in os.walk(folder):
        for name in sorted(names):
            path = os.path.join(root, name)
            ext = os.path.splitext(name.lower())[1]

            if os.path.isfile(path) and ext in IMAGE_EXTENSIONS:
                files.append(path)

    return sorted(files)


def _make_meta(
    preprocess_mode: str,
    enhancement_mode: str,
    mean_threshold: float,
    p90_threshold: float,
    threshold_mode: str,
):
    """
    Метаданные базы.

    Зачем нужны:
    если база построена в одном режиме, а query анализируется в другом,
    результаты будут некорректными. Поэтому режимы сохраняются в db.json.
    """
    return {
        "preprocess_mode": preprocess_mode,
        "enhancement_mode": enhancement_mode,
        "mean_threshold": float(mean_threshold),
        "p90_threshold": float(p90_threshold),
        "threshold_mode": threshold_mode,
    }


def build_database(
    reference_folder: str,
    db_path: str,
    plots_folder: str | None = None,
    preprocess_mode: str = "none",
    enhancement_mode: str = "gamma_then_clahe",
    mean_threshold: float = 105.0,
    p90_threshold: float = 170.0,
    threshold_mode: str = "dense",
):
    """
    Строит базу эталонных ROI.

    В базе хранятся НЕ изображения, а признаки:
    - имя ROI;
    - путь к исходному файлу;
    - числовой vector;
    - bars для отладки и визуализации.

    threshold_mode:
    - dense    — старый режим v0.2;
    - fixed    — фиксированные пороги;
    - quantile — адаптивные процентильные пороги;
    - hybrid   — экспериментальный смешанный режим.
    """
    records = []

    if plots_folder:
        os.makedirs(plots_folder, exist_ok=True)

    image_paths = list_images(reference_folder)

    if not image_paths:
        raise FileNotFoundError(
            f"В папке reference_folder нет изображений: {reference_folder}"
        )

    for image_path in image_paths:
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

        vector = barcode_to_vector(
            bars,
            gray=gray,
            threshold_mode=threshold_mode,
        ).tolist()

        name = os.path.splitext(os.path.basename(image_path))[0]

        records.append({
            "name": name,
            "image_path": image_path,
            "vector": vector,
            "bars": bars,
        })

        if plots_folder:
            plot_barcode(
                bars,
                os.path.join(plots_folder, f"{name}_barcode.png"),
                title=f"{name} | {threshold_mode}",
            )

    payload = {
        "meta": _make_meta(
            preprocess_mode=preprocess_mode,
            enhancement_mode=enhancement_mode,
            mean_threshold=mean_threshold,
            p90_threshold=p90_threshold,
            threshold_mode=threshold_mode,
        ),
        "records": records,
    }

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"База построена: {db_path}")
    print(f"Количество записей: {len(records)}")
    print(f"threshold_mode: {threshold_mode}")


def _load_db_payload(db_path: str):
    """
    Загружает db.json.

    Поддерживает два варианта:
    1. старый формат: список records;
    2. новый формат: {"meta": ..., "records": ...}.
    """
    with open(db_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Старый формат: просто список
    if isinstance(payload, list):
        return {}, payload

    # Новый формат
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError(f"Некорректный формат db файла: {db_path}")

    meta = payload.get("meta", {})
    records = payload["records"]

    return meta, records


def _check_db_meta_compatibility(
    db_meta: dict,
    preprocess_mode: str,
    enhancement_mode: str,
    mean_threshold: float,
    p90_threshold: float,
    threshold_mode: str,
):
    """
    Проверяет, что база и query считаются в одинаковых режимах.

    Это важно:
    dense и quantile могут давать векторы разной длины и разной природы.
    """
    if not db_meta:
        # Старые базы без meta считаем legacy.
        # Для них совместимость не проверяем.
        return

    mismatches = []

    if db_meta.get("preprocess_mode") != preprocess_mode:
        mismatches.append(
            f"preprocess_mode: db={db_meta.get('preprocess_mode')} vs query={preprocess_mode}"
        )

    if db_meta.get("enhancement_mode") != enhancement_mode:
        mismatches.append(
            f"enhancement_mode: db={db_meta.get('enhancement_mode')} vs query={enhancement_mode}"
        )

    if float(db_meta.get("mean_threshold", mean_threshold)) != float(mean_threshold):
        mismatches.append(
            f"mean_threshold: db={db_meta.get('mean_threshold')} vs query={mean_threshold}"
        )

    if float(db_meta.get("p90_threshold", p90_threshold)) != float(p90_threshold):
        mismatches.append(
            f"p90_threshold: db={db_meta.get('p90_threshold')} vs query={p90_threshold}"
        )

    # Старые новые-базы могли не иметь threshold_mode.
    # По умолчанию считаем их dense, потому что это старый barcode.py.
    db_threshold_mode = db_meta.get("threshold_mode", "dense")

    if db_threshold_mode != threshold_mode:
        mismatches.append(
            f"threshold_mode: db={db_threshold_mode} vs query={threshold_mode}"
        )

    if mismatches:
        raise ValueError(
            "Несовпадение режима между базой и query:\n- "
            + "\n- ".join(mismatches)
            + "\nПострой базу заново или используй те же параметры."
        )


def _check_vector_lengths(query_vector: np.ndarray, db_records: list):
    """
    Проверяет, что длина query-вектора совпадает с длиной векторов в базе.

    Это дополнительная защита от неправильного сравнения dense/quantile.
    """
    if not db_records:
        raise ValueError("База пуста: нет records.")

    query_len = len(query_vector)

    for rec in db_records:
        ref_len = len(rec.get("vector", []))

        if ref_len != query_len:
            raise ValueError(
                f"Несовпадение длины векторов: query={query_len}, "
                f"reference={ref_len}, record={rec.get('name')}. "
                f"Скорее всего база построена в другом threshold_mode."
            )


def match_against_database(
    query_image_path: str,
    db_path: str,
    preprocess_mode: str = "none",
    enhancement_mode: str = "gamma_then_clahe",
    mean_threshold: float = 105.0,
    p90_threshold: float = 170.0,
    threshold_mode: str = "dense",
):
    """
    Сравнивает query ROI с базой эталонных ROI.

    Возвращает:
    - query_bars;
    - отсортированный список результатов.

    Чем меньше distance, тем ближе эталон.
    similarity = 1 / (1 + distance).
    """
    db_meta, db_records = _load_db_payload(db_path)

    _check_db_meta_compatibility(
        db_meta=db_meta,
        preprocess_mode=preprocess_mode,
        enhancement_mode=enhancement_mode,
        mean_threshold=mean_threshold,
        p90_threshold=p90_threshold,
        threshold_mode=threshold_mode,
    )

    gray = load_roi(
        query_image_path,
        preprocess_mode=preprocess_mode,
        enhancement_mode=enhancement_mode,
        mean_threshold=mean_threshold,
        p90_threshold=p90_threshold,
    )

    query_bars = build_topo_barcode(
        gray,
        threshold_mode=threshold_mode,
    )

    query_vector = barcode_to_vector(
        query_bars,
        gray=gray,
        threshold_mode=threshold_mode,
    )

    _check_vector_lengths(query_vector, db_records)

    results = []

    for rec in db_records:
        ref_vector = np.array(rec["vector"], dtype=np.float32)

        distance = float(np.linalg.norm(query_vector - ref_vector))
        similarity = 1.0 / (1.0 + distance)

        results.append({
            "name": rec["name"],
            "distance": round(distance, 6),
            "similarity": round(similarity, 6),
            "image_path": rec.get("image_path", ""),
        })

    results.sort(key=lambda x: x["distance"])

    return query_bars, results