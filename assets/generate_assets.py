"""
generate_assets.py
Run this script ONCE to create the three pre-loaded target images.

    python assets/generate_assets.py

Produces:
    assets/josephs_logo.png     — colourful geometric logo
    assets/simple_shape.png     — minimal geometric target (easy)
    assets/complex_icon.png     — busy multi-colour target (hard)
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

SIZE = 128
OUT = Path(__file__).parent


def _save(img: Image.Image, name: str) -> None:
    path = OUT / name
    img.save(path)
    print(f"  saved -> {path}")


# ---------------------------------------------------------------------------
# 1. josephs_logo.png  — bold overlapping triangles forming a "J" motif
# ---------------------------------------------------------------------------
def make_josephs_logo() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), (15, 15, 40))
    draw = ImageDraw.Draw(img, "RGBA")

    shapes = [
        # background large triangle
        [(10, 10), (118, 10), (64, 118)],
        # gold accent
        [(20, 20), (108, 20), (64, 100)],
        # left wing
        [(5, 60), (55, 10), (55, 110)],
        # right wing
        [(123, 60), (73, 10), (73, 110)],
        # centre highlight
        [(44, 44), (84, 44), (64, 90)],
    ]
    colors = [
        (70, 130, 180, 220),
        (255, 200, 50, 200),
        (220, 80, 60, 180),
        (80, 200, 120, 180),
        (255, 255, 255, 160),
    ]
    for pts, col in zip(shapes, colors):
        draw.polygon(pts, fill=col)
    return img


# ---------------------------------------------------------------------------
# 2. simple_shape.png  — solid coloured circle + two triangles (easy target)
# ---------------------------------------------------------------------------
def make_simple_shape() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), (240, 240, 240))
    draw = ImageDraw.Draw(img, "RGBA")

    # Blue filled circle
    draw.ellipse([30, 30, 98, 98], fill=(50, 100, 220))

    # Red triangle top-right
    draw.polygon([(80, 10), (118, 10), (118, 50)], fill=(220, 60, 60, 200))

    # Green triangle bottom-left
    draw.polygon([(10, 80), (50, 118), (10, 118)], fill=(60, 200, 80, 200))
    return img


# ---------------------------------------------------------------------------
# 3. complex_icon.png  — dense multi-coloured mosaic (challenging target)
# ---------------------------------------------------------------------------
def make_complex_icon() -> Image.Image:
    rng = np.random.default_rng(42)
    img = Image.new("RGB", (SIZE, SIZE), (20, 20, 20))
    draw = ImageDraw.Draw(img, "RGBA")

    for _ in range(60):
        xs = rng.integers(0, SIZE, 3).tolist()
        ys = rng.integers(0, SIZE, 3).tolist()
        pts = list(zip(xs, ys))
        r, g, b = rng.integers(30, 256, 3).tolist()
        a = rng.integers(120, 220)
        draw.polygon(pts, fill=(r, g, b, a))
    return img


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating asset images …")
    _save(make_josephs_logo(), "josephs_logo.png")
    _save(make_simple_shape(), "simple_shape.png")
    _save(make_complex_icon(), "complex_icon.png")
    print("Done.")
