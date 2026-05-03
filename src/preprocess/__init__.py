from .common import read_base_steps
from .mode_none import run_mode_none
from .mode_always import run_mode_always
from .mode_auto import run_mode_auto


def preprocess_steps(
    image_path: str,
    size=(128, 128),
    blur_ksize: int = 3,
    preprocess_mode: str = "none",   # none / always / auto
    enhancement_mode: str = "gamma_then_clahe",
    mean_threshold: float = 105.0,
    p90_threshold: float = 170.0,
):
    base_steps = read_base_steps(image_path=image_path, size=size)

    if preprocess_mode == "none":
        return run_mode_none(base_steps, blur_ksize=blur_ksize)

    elif preprocess_mode == "always":
        return run_mode_always(
            base_steps,
            blur_ksize=blur_ksize,
            enhancement_mode=enhancement_mode
        )

    elif preprocess_mode == "auto":
        return run_mode_auto(
            base_steps,
            blur_ksize=blur_ksize,
            enhancement_mode=enhancement_mode,
            mean_threshold=mean_threshold,
            p90_threshold=p90_threshold
        )

    else:
        raise ValueError(
            f"Неизвестный preprocess_mode: {preprocess_mode}. "
            f"Ожидается: none / always / auto"
        )


def load_roi(
    image_path: str,
    size=(128, 128),
    blur_ksize: int = 3,
    preprocess_mode: str = "none",
    enhancement_mode: str = "gamma_then_clahe",
    mean_threshold: float = 105.0,
    p90_threshold: float = 170.0,
):
    steps = preprocess_steps(
        image_path=image_path,
        size=size,
        blur_ksize=blur_ksize,
        preprocess_mode=preprocess_mode,
        enhancement_mode=enhancement_mode,
        mean_threshold=mean_threshold,
        p90_threshold=p90_threshold,
    )
    return steps["gray_final"]