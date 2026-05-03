import numpy as np


def _to_int_thresholds(values):
    """
    Приводим пороги к int и ограничиваем диапазон 1..254.
    Важно: НЕ удаляем дубли, потому что длина вектора должна быть одинаковой
    для базы и для query.
    """
    thresholds = []

    for v in values:
        t = int(round(float(v)))
        t = max(1, min(254, t))
        thresholds.append(t)

    return thresholds


def fixed_thresholds(gray=None):
    """
    Фиксированные пороги.
    Быстро, но хуже работает при bright/dark.
    """
    return [80, 120, 160]


def quantile_thresholds(gray):
    """
    Адаптивные пороги по процентилям яркости.
    Всегда возвращает ровно 4 значения: p50, p65, p80, p90.
    Дубли НЕ удаляются.
    """
    if gray is None:
        raise ValueError("Для threshold_mode='quantile' нужно передать gray.")

    values = np.percentile(gray, [50, 65, 80, 90])
    return _to_int_thresholds(values)


def dense_thresholds(gray=None):
    """
    Плотная фиксированная сетка порогов.
    """
    return list(range(240, 15, -16))


def hybrid_thresholds(gray):
    """
    Гибрид: фиксированные + квантильные.
    Чтобы длина была стабильной, тоже НЕ удаляем дубли.
    """
    return fixed_thresholds(gray) + quantile_thresholds(gray)


def get_thresholds(gray=None, mode: str = "dense"):
    """
    Единая функция выбора порогов.
    mode:
      - fixed
      - dense
      - quantile
      - hybrid
    """
    mode = (mode or "dense").lower().strip()

    if mode == "fixed":
        return fixed_thresholds(gray)

    if mode == "dense":
        return dense_thresholds(gray)

    if mode == "quantile":
        return quantile_thresholds(gray)

    if mode == "hybrid":
        return hybrid_thresholds(gray)

    raise ValueError(
        f"Неизвестный threshold_mode: {mode}. "
        f"Доступно: fixed, dense, quantile, hybrid"
    )