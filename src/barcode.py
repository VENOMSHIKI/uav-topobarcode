import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.thresholds import get_thresholds


def default_thresholds():
    """
    Старый базовый режим v0.2:
    плотная сетка фиксированных порогов от светлого к темному.

    Важно:
    пороги идут по убыванию, потому что build_topo_barcode()
    считает persistence = birth - death.
    """
    return list(range(240, 15, -16))  # 240, 224, ..., 32


def resolve_thresholds(gray: np.ndarray, thresholds=None, threshold_mode: str = "dense"):
    """
    Единая точка выбора порогов для barcode.py.

    threshold_mode:
    - dense    — старый режим v0.2: 240, 224, ..., 32;
    - fixed    — простые фиксированные пороги из thresholds.py: 80, 120, 160;
    - quantile — адаптивные пороги по процентилям яркости ROI;
    - hybrid   — fixed + quantile.

    Для топологического баркода пороги всегда сортируются по убыванию.
    """
    if thresholds is not None:
        result = list(thresholds)
    else:
        mode = str(threshold_mode).lower().strip()

        if mode == "dense":
            result = default_thresholds()
        else:
            result = get_thresholds(gray, mode)

    # Для текущей логики barcode.py нужен порядок от высокого порога к низкому.
    result = sorted([int(t) for t in result], reverse=True)

    return result


def extract_components(binary: np.ndarray, min_area: int = 20):
    """
    Выделение связных компонент на бинарном изображении.

    binary:
    - черно-белое изображение после threshold;
    - белые области считаются foreground.

    min_area:
    - отсекает слишком мелкий шум.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    components = []

    for label in range(1, n_labels):  # 0 = фон
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        mask = (labels == label).astype(np.uint8)

        components.append({
            "mask": mask,
            "area": area
        })

    return components


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    IoU двух масок.

    Нужно для грубого отслеживания компоненты между соседними порогами:
    если область на текущем пороге сильно пересекается с областью на прошлом,
    считаем, что это продолжение той же компоненты.
    """
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()

    return float(inter) / float(union) if union > 0 else 0.0


def compute_threshold_curves(
    gray: np.ndarray,
    thresholds=None,
    min_area: int = 20,
    threshold_mode: str = "dense"
):
    """
    Считает легкие кривые по порогам:
    - количество компонент;
    - доля foreground-площади.

    Эти признаки добавляются к баркод-вектору.
    """
    thresholds = resolve_thresholds(
        gray,
        thresholds=thresholds,
        threshold_mode=threshold_mode
    )

    comp_counts = []
    area_ratios = []

    image_area = gray.shape[0] * gray.shape[1]

    for t in thresholds:
        _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        comps = extract_components(binary, min_area=min_area)

        comp_counts.append(len(comps))

        total_area = sum(c["area"] for c in comps)
        area_ratios.append(min(total_area / float(image_area), 1.0))

    return (
        np.array(comp_counts, dtype=np.float32),
        np.array(area_ratios, dtype=np.float32),
    )


def build_topo_barcode(
    gray: np.ndarray,
    thresholds=None,
    min_area: int = 20,
    match_iou: float = 0.15,
    threshold_mode: str = "dense"
):
    """
    Строит топологический barcode ROI.

    Логика:
    1. Берем grayscale ROI.
    2. Идем по набору порогов яркости.
    3. На каждом пороге строим бинарное изображение.
    4. Выделяем компоненты связности.
    5. Отслеживаем компоненты между соседними порогами через IoU.
    6. Получаем интервалы существования компонент: birth/death.
    """
    thresholds = resolve_thresholds(
        gray,
        thresholds=thresholds,
        threshold_mode=threshold_mode
    )

    tracks = {}
    next_track_id = 0
    prev_components = []

    for t in thresholds:
        _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        current_components = extract_components(binary, min_area=min_area)

        used_prev = set()

        for comp in current_components:
            best_idx = None
            best_score = 0.0

            for idx, prev in enumerate(prev_components):
                if idx in used_prev:
                    continue

                score = mask_iou(comp["mask"], prev["mask"])

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None and best_score >= match_iou:
                track_id = prev_components[best_idx]["track_id"]
                used_prev.add(best_idx)

                tracks[track_id]["death"] = t
                tracks[track_id]["max_area"] = max(
                    tracks[track_id]["max_area"],
                    comp["area"]
                )

                comp["track_id"] = track_id
            else:
                track_id = next_track_id
                next_track_id += 1

                tracks[track_id] = {
                    "track_id": track_id,
                    "birth": t,
                    "death": t,
                    "max_area": comp["area"]
                }

                comp["track_id"] = track_id

        prev_components = current_components

    bars = []

    for tr in tracks.values():
        persistence = tr["birth"] - tr["death"]

        bars.append({
            "track_id": tr["track_id"],
            "birth": int(tr["birth"]),
            "death": int(tr["death"]),
            "persistence": int(persistence),
            "max_area": int(tr["max_area"])
        })

    bars.sort(key=lambda x: (x["persistence"], x["max_area"]), reverse=True)

    return bars


def barcode_to_vector(
    bars,
    gray: np.ndarray,
    thresholds=None,
    top_k: int = 12,
    min_area: int = 20,
    threshold_mode: str = "dense"
):
    """
    Переводит barcode в числовой вектор.

    Что входит в вектор:
    1. top-K самых устойчивых баров:
       birth, death, persistence, max_area.
    2. Кривые по порогам:
       количество компонент и доля foreground-площади.
    """
    thresholds = resolve_thresholds(
        gray,
        thresholds=thresholds,
        threshold_mode=threshold_mode
    )

    image_area = gray.shape[0] * gray.shape[1]
    vec = []

    # Базовая часть: top-K баров
    for bar in bars[:top_k]:
        vec.extend([
            bar["birth"] / 255.0,
            bar["death"] / 255.0,
            bar["persistence"] / 255.0,
            min(bar["max_area"] / float(image_area), 1.0)
        ])

    while len(vec) < top_k * 4:
        vec.append(0.0)

    # Дополнительная часть: легкие кривые по порогам
    comp_counts, area_ratios = compute_threshold_curves(
        gray,
        thresholds=thresholds,
        min_area=min_area,
        threshold_mode=threshold_mode
    )

    comp_counts_norm = np.clip(comp_counts / 20.0, 0.0, 1.0)
    area_ratios_norm = np.clip(area_ratios, 0.0, 1.0)

    vec.extend(comp_counts_norm.tolist())
    vec.extend(area_ratios_norm.tolist())

    return np.array(vec, dtype=np.float32)


def plot_barcode(
    bars,
    out_path: str,
    title: str = "Topological barcode"
):
    """
    Сохраняет картинку barcode.
    """
    plt.figure(figsize=(8, 4))

    for i, bar in enumerate(bars):
        plt.hlines(
            y=i,
            xmin=bar["death"],
            xmax=bar["birth"],
            linewidth=2
        )

    plt.xlabel("Threshold")
    plt.ylabel("Bar index")
    plt.title(title)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_threshold_curves(
    gray: np.ndarray,
    out_path: str,
    thresholds=None,
    min_area: int = 20,
    title: str = "Threshold curves",
    threshold_mode: str = "dense"
):
    """
    Сохраняет графики:
    - количество компонент по порогам;
    - foreground area ratio по порогам.
    """
    thresholds = resolve_thresholds(
        gray,
        thresholds=thresholds,
        threshold_mode=threshold_mode
    )

    comp_counts, area_ratios = compute_threshold_curves(
        gray,
        thresholds=thresholds,
        min_area=min_area,
        threshold_mode=threshold_mode
    )

    fig = plt.figure(figsize=(8, 5))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(thresholds, comp_counts, marker="o")
    ax1.set_title("Количество компонент по порогам")
    ax1.set_ylabel("Count")
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(thresholds, area_ratios, marker="o")
    ax2.set_title("Доля foreground-площади по порогам")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Area ratio")
    ax2.invert_xaxis()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()