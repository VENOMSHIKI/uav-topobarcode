from .common import apply_blur
from .enhance import enhance_image


def run_mode_always(
    base_steps: dict,
    blur_ksize: int = 3,
    enhancement_mode: str = "gamma_then_clahe"
) -> dict:
    gray_enhanced = enhance_image(
        base_steps["gray_norm"],
        enhancement_mode=enhancement_mode
    )
    gray_final = apply_blur(gray_enhanced, blur_ksize=blur_ksize)

    out = dict(base_steps)
    out.update({
        "gray_enhanced": gray_enhanced,
        "gray_final": gray_final,
        "is_dark": True,
        "enhancement_applied": True,
        "preprocess_mode": "always",
        "enhancement_mode": enhancement_mode,
    })
    return out