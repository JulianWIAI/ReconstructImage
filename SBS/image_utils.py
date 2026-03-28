"""
image_utils.py
Image loading, caching, fast NumPy rendering, and fitness calculation.

Performance design:
  - render_genome_fast(): Pure NumPy barycentric triangle rasteriser.
    Eliminates ALL per-triangle Pillow Image allocations from the
    original pipeline.  A pre-allocated pixel meshgrid is cached at
    module level so it is only computed once per image resolution.

  - compute_fitness_weighted(): Vectorised MSE with an edge-importance
    weight map — boundary pixels contribute more to the loss, forcing
    the EA to prioritise the JOSEPHS ring contours.

  - compute_edge_map(): Sobel-based gradient magnitude in pure NumPy.

  - Dual-resolution strategy: the optimiser uses a small (e.g. 128×128)
    downsampled image for fitness; the Streamlit dashboard shows the
    original full-resolution image for visual quality.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from PIL import Image
from typing import TYPE_CHECKING, Tuple, Dict

if TYPE_CHECKING:
    from SBS.dna_structures import Genome


# ---------------------------------------------------------------------------
# Module-level pixel grid cache
# _PIXEL_GRID_CACHE : (h, w) → (px, py) float32 meshgrids — IMMUTABLE after
#   creation, so safe to share across Streamlit threads/sessions.
#
# NOTE: We intentionally do NOT cache mutable workspace arrays (e0, e1, inside
#   etc.).  Streamlit can run multiple script executions in different threads
#   sharing the same module globals.  Mutable shared arrays cause data races
#   that corrupt boolean masks mid-blending, producing the "shape mismatch"
#   IndexError seen at runtime.  Local variables per call are the only safe
#   option; the allocation cost at 128×128 is negligible.
# ---------------------------------------------------------------------------
_PIXEL_GRID_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}


def _get_pixel_grid(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return cached read-only (px, py) meshgrids for an (h, w) canvas.
    These arrays are never modified after creation, so caching is thread-safe.
    """
    key = (h, w)
    if key not in _PIXEL_GRID_CACHE:
        py, px = np.mgrid[0:h, 0:w]
        _PIXEL_GRID_CACHE[key] = (
            px.astype(np.float32),
            py.astype(np.float32),
        )
    return _PIXEL_GRID_CACHE[key]


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_and_resize(image_path: str, target_size: int = 128) -> np.ndarray:
    """
    Load an image from disk, convert to RGB, resize to a square canvas.
    Returns a uint8 NumPy array of shape (H, W, 3).
    Used for the low-resolution fitness target.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def load_full_res(image_path: str, max_display_size: int = 512) -> np.ndarray:
    """
    Load the original image, capped at max_display_size on the longest edge.
    Used for high-quality dashboard display only (not fitness calculation).
    Returns a uint8 NumPy array of shape (H, W, 3).
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_display_size / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def list_asset_images(assets_dir: str) -> list:
    """Return a sorted list of image file names in the assets directory."""
    assets_path = Path(assets_dir)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    return sorted(
        p.name
        for p in assets_path.iterdir()
        if p.suffix.lower() in extensions
    )


# ---------------------------------------------------------------------------
# Edge map (Sobel gradient, pure NumPy)
# ---------------------------------------------------------------------------

def compute_edge_map(target_rgb: np.ndarray) -> np.ndarray:
    """
    Compute a normalised (H, W) float32 edge-importance map via Sobel.

    High values mark pixels at colour boundaries (e.g. the red/blue ring
    edges of the JOSEPHS logo).  Used for:
      1. Weighted MSE fitness (boundary pixels count more).
      2. Edge-biased vertex initialisation in the EA.

    Pure NumPy — no scipy/skimage dependency required.
    """
    # Convert to float32 luminance  [0, 1]
    lum = (
        0.299 * target_rgb[:, :, 0]
        + 0.587 * target_rgb[:, :, 1]
        + 0.114 * target_rgb[:, :, 2]
    ).astype(np.float32) / 255.0

    # Horizontal Sobel  (3×3, approximated with 1D separable ops)
    # Gy = [[-1,-2,-1],[0,0,0],[1,2,1]] applied via two 1D passes
    smooth_h = lum[:, :-2] + 2 * lum[:, 1:-1] + lum[:, 2:]   # horizontal blur row
    gx = smooth_h[2:, :] - smooth_h[:-2, :]                   # vertical diff

    smooth_v = lum[:-2, :] + 2 * lum[1:-1, :] + lum[2:, :]   # vertical blur col
    gy = smooth_v[:, 2:] - smooth_v[:, :-2]                   # horizontal diff

    # Pad back to original size
    h, w = lum.shape
    grad = np.zeros((h, w), dtype=np.float32)
    # gx covers rows [1:-1], cols [0 : w-2]
    grad[1:-1, 0 : w - 2] += np.abs(gx)
    # gy covers rows [0 : h-2], cols [1:-1]
    grad[0 : h - 2, 1:-1] += np.abs(gy)

    # Normalise to [0, 1]
    max_val = grad.max()
    if max_val > 1e-8:
        grad /= max_val

    return grad


def get_high_edge_coords(edge_map: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Return an (K, 2) int array of [row, col] coordinates where the edge
    map exceeds `threshold`.  Used for edge-biased vertex initialisation.
    """
    ys, xs = np.where(edge_map > threshold)
    if len(ys) == 0:
        # Fallback: all pixels
        h, w = edge_map.shape
        ys, xs = np.mgrid[0:h, 0:w]
        ys, xs = ys.ravel(), xs.ravel()
    return np.stack([ys, xs], axis=1)  # (K, 2)


# ---------------------------------------------------------------------------
# Fast NumPy renderer (barycentric rasterisation)
# ---------------------------------------------------------------------------

def render_genome_fast(dna: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Rasterise a genome into a uint8 RGB array using pure NumPy.

    Algorithm
    ---------
    For each triangle (row in dna):
      1. Compute three edge-function values for every pixel (H×W) using
         NumPy broadcasting — no Python pixel loop.
      2. A pixel is inside the triangle when all three edge values share
         the same sign (handles both CW and CCW vertex winding).
      3. Apply Porter-Duff source-over alpha blending on the masked pixels.

    The pixel meshgrid (px, py) is pre-computed and cached at module
    level, so it is created only once per resolution.

    Parameters
    ----------
    dna : (N, 10) float32
        [x0, y0, x1, y1, x2, y2, r, g, b, alpha] per triangle.
    w, h : int
        Canvas dimensions in pixels.

    Returns
    -------
    np.ndarray  (H, W, 3) uint8
    """
    px, py = _get_pixel_grid(h, w)   # read-only cached (H, W) float32
    canvas = np.zeros((h, w, 3), dtype=np.float32)

    # Per-channel views: shape (H, W).  Assignments via a 2D boolean mask on
    # a 2D array are unambiguous — avoids multi-dim advanced indexing pitfalls.
    ch_r = canvas[:, :, 0]
    ch_g = canvas[:, :, 1]
    ch_b = canvas[:, :, 2]

    for row in dna:
        x0, y0 = row[0], row[1]
        x1, y1 = row[2], row[3]
        x2, y2 = row[4], row[5]
        r, g, b = row[6], row[7], row[8]
        alpha   = row[9]

        # ---- Barycentric edge functions (all H×W pixels at once) -----------
        # e_i = (px - xi)*(yj - yi) - (py - yi)*(xj - xi)
        # Local variables only — no shared mutable state, thread-safe.
        e0 = (px - x0) * (y1 - y0) - (py - y0) * (x1 - x0)
        e1 = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        e2 = (px - x2) * (y0 - y2) - (py - y2) * (x0 - x2)

        # Pixel is inside when all three edge values share the same sign
        # (handles both CW and CCW vertex winding orders).
        inside: np.ndarray = (
            ((e0 >= 0.0) & (e1 >= 0.0) & (e2 >= 0.0))
            | ((e0 <= 0.0) & (e1 <= 0.0) & (e2 <= 0.0))
        )

        if not inside.any():
            continue

        # ---- Porter-Duff source-over blending on masked pixels -------------
        # dst_new = alpha * src + (1 - alpha) * dst_old
        # Using per-channel 2D views + 2D boolean mask: unambiguous indexing.
        oma = 1.0 - alpha
        ch_r[inside] = alpha * r + oma * ch_r[inside]
        ch_g[inside] = alpha * g + oma * ch_g[inside]
        ch_b[inside] = alpha * b + oma * ch_b[inside]

    return np.clip(canvas, 0.0, 255.0).astype(np.uint8)


def genome_to_pil(genome: "Genome") -> Image.Image:
    """Render a genome and return a PIL Image for Streamlit display."""
    arr = render_genome_fast(genome.dna, genome.img_width, genome.img_height)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Fitness functions
# ---------------------------------------------------------------------------

def compute_fitness(rendered: np.ndarray, target_f32: np.ndarray) -> float:
    """
    Vectorised Mean Squared Error.

    Parameters
    ----------
    rendered   : (H, W, 3) uint8
    target_f32 : (H, W, 3) float32  — pre-cast for speed (cast once, reuse many times)

    Returns
    -------
    float  MSE in [0, 65025]; lower is better.
    """
    # Cast once per call — target_f32 is pre-cast by the caller
    diff = rendered.astype(np.float32) - target_f32
    # Vectorised: diff**2 then mean — zero Python pixel loops
    return float(np.mean(diff * diff))


def compute_fitness_weighted(
    rendered: np.ndarray,
    target_f32: np.ndarray,
    edge_weights: np.ndarray,
) -> float:
    """
    Edge-weighted MSE.

    Pixels at colour boundaries (high edge_weights) contribute up to
    4× more to the fitness score.  This forces the EA to prioritise
    reconstructing the sharp rings of the JOSEPHS logo before worrying
    about flat interior regions.

    Parameters
    ----------
    rendered      : (H, W, 3) uint8
    target_f32    : (H, W, 3) float32
    edge_weights  : (H, W)    float32, normalised [0, 1]

    Returns
    -------
    float  weighted MSE; lower is better.
    """
    diff_sq = (rendered.astype(np.float32) - target_f32) ** 2  # (H, W, 3)

    # Expand edge weights to (H, W, 1) for broadcasting over 3 channels
    w = (1.0 + 3.0 * edge_weights)[:, :, np.newaxis]           # (H, W, 1)

    # Normalised weighted mean
    return float(np.sum(diff_sq * w) / (np.sum(w) * 3.0))
