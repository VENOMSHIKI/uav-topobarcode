from .common import apply_blur


def run_mode_none(base_steps: dict, blur_ksize: int = 3) -> dict:
    gray_enhanced = base_steps["gray_norm"].copy()
    gray_final = apply_blur(gray_enhanced, blur_ksize=blur_ksize)

    out = dict(base_steps)
    out.update({
        "gray_enhanced": gray_enhanced,
        "gray_final": gray_final,
        "is_dark": False,
        "enhancement_applied": False,
        "preprocess_mode": "none",
        "enhancement_mode": None,
    })
    return out