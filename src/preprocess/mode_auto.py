from .common import apply_blur
from .enhance import enhance_image


def decide_is_dark(
    base_steps: dict,
    mean_threshold: float = 105.0,
    p90_threshold: float = 170.0
) -> bool:
    """
    ROI считаем темным, если:
    - средняя яркость низкая
    - и верхний яркостный хвост тоже не очень высокий

    Это лучше, чем смотреть только на mean.
    """
    mean_raw = base_steps["mean_brightness_raw"]
    p90_raw = base_steps["p90_raw"]

    return (mean_raw < mean_threshold) and (p90_raw < p90_threshold)


def run_mode_auto(
    base_steps: dict,
    blur_ksize: int = 3,
    enhancement_mode: str = "gamma_then_clahe",
    mean_threshold: float = 105.0,
    p90_threshold: float = 170.0
) -> dict:
    is_dark = decide_is_dark(
        base_steps,
        mean_threshold=mean_threshold,
        p90_threshold=p90_threshold
    )

    if is_dark:
        gray_enhanced = enhance_image(
            base_steps["gray_norm"],
            enhancement_mode=enhancement_mode
        )
        enhancement_applied = True
    else:
        gray_enhanced = base_steps["gray_norm"].copy()
        enhancement_applied = False

    gray_final = apply_blur(gray_enhanced, blur_ksize=blur_ksize)

    out = dict(base_steps)
    out.update({
        "gray_enhanced": gray_enhanced,
        "gray_final": gray_final,
        "is_dark": is_dark,
        "enhancement_applied": enhancement_applied,
        "preprocess_mode": "auto",
        "enhancement_mode": enhancement_mode,
        "mean_threshold": mean_threshold,
        "p90_threshold": p90_threshold,
    })
    return out