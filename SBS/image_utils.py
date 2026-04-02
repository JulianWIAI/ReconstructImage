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
    the EA to prioritise the target's ring contours.

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
    Compute a normalised (H, W) float32 edge-importance map via Sobel filtering.

    The Sobel operator approximates the image gradient magnitude — a measure
    of how rapidly pixel intensity changes in each direction.  Pixels at colour
    boundary pixels (e.g. the sharp red-to-white ring edge of the target image) have
    a high gradient magnitude; flat interior regions have a gradient near zero.

    The resulting map is used for two purposes:
      1. Weighted MSE fitness: boundary pixels penalise the EA up to 4× more,
         forcing triangle placement to align with sharp edges first.
      2. Edge-biased vertex initialisation: newly created or burst-regenerated
         genomes have some vertices snapped near high-gradient pixels, injecting
         domain knowledge about where the important features are located.

    Implementation: pure NumPy separable 1D passes approximate the full 3×3
    Sobel kernel without scipy or skimage — keeping the dependency list minimal.

    Parameters
    ----------
    target_rgb : (H, W, 3) uint8

    Returns
    -------
    np.ndarray  (H, W) float32, normalised to [0, 1].
    """
    # Convert to float32 luminance [0, 1] using standard ITU-R BT.601 weights.
    # A single-channel luminance image is sufficient — the Sobel gradient detects
    # brightness changes which correspond to colour boundary edges.
    lum = (
        0.299 * target_rgb[:, :, 0]
        + 0.587 * target_rgb[:, :, 1]
        + 0.114 * target_rgb[:, :, 2]
    ).astype(np.float32) / 255.0

    # Sobel operator decomposed into separable 1D passes (more efficient than
    # a 2D convolution for a 3×3 kernel):
    #   Gx ≈ [-1,0,1] (horizontal) convolved with [1,2,1]ᵀ (vertical blur)
    #   Gy ≈ [1,2,1]  (vertical)  convolved with [-1,0,1]ᵀ (horizontal blur)
    # The blurring step suppresses noise before differencing, which is why
    # Sobel outperforms a plain finite-difference gradient on real images.
    smooth_h = lum[:, :-2] + 2 * lum[:, 1:-1] + lum[:, 2:]   # row-wise blur
    gx = smooth_h[2:, :] - smooth_h[:-2, :]                   # vertical diff → Gx

    smooth_v = lum[:-2, :] + 2 * lum[1:-1, :] + lum[2:, :]   # col-wise blur
    gy = smooth_v[:, 2:] - smooth_v[:, :-2]                   # horizontal diff → Gy

    # Accumulate gradient magnitudes back into a full-size canvas.
    # Using |Gx| + |Gy| as an L1 approximation to √(Gx² + Gy²) avoids a sqrt
    # while preserving the relative ordering of edge strength — sufficient here.
    h, w = lum.shape
    grad = np.zeros((h, w), dtype=np.float32)
    grad[1:-1, 0 : w - 2] += np.abs(gx)
    grad[0 : h - 2, 1:-1] += np.abs(gy)

    # Normalise to [0, 1] so the weight map is resolution-independent.
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
    Unweighted Mean Squared Error between a rendered genome and the target image.

    MSE = (1 / (H × W × 3)) × Σ (rendered_i − target_i)²

    Squaring the per-pixel differences penalises large errors more than small
    ones (a pixel that is 20 units off contributes 4× as much as one that is
    10 units off), which encourages the EA to eliminate gross mismatches before
    refining subtle ones.  The theoretical maximum for uint8 RGB is 255² = 65025.

    Parameters
    ----------
    rendered   : (H, W, 3) uint8
    target_f32 : (H, W, 3) float32  — pre-cast by the caller to avoid
                  repeated type conversion in the hot evaluation loop.

    Returns
    -------
    float  MSE in [0, 65025]; lower is better.
    """
    # Cast rendered once; target_f32 is pre-cast by the caller (cast once,
    # reuse across the entire population evaluation — significant speed saving).
    diff = rendered.astype(np.float32) - target_f32
    return float(np.mean(diff * diff))


def compute_fitness_weighted(
    rendered: np.ndarray,
    target_f32: np.ndarray,
    edge_weights: np.ndarray,
) -> float:
    """
    Edge-weighted Mean Squared Error — the primary fitness function used by the EA.

    Formula
    -------
    weighted_MSE = Σ [ diff²(x,y,c) × w(x,y) ]  /  [ Σ w(x,y) × 3 ]

    where  w(x,y) = 1 + 3 × edge_map(x,y)

    Why edge weighting?
    -------------------
    A plain MSE treats every pixel equally: a flat background patch and a
    sharp ring edge contribute the same amount to the loss.  This causes the
    EA to waste many generations perfecting large, low-frequency areas while
    ignoring the fine edges that humans perceive as the most defining feature
    of the target image.

    The weight map w(x,y) linearly amplifies the contribution of boundary
    pixels: an interior pixel (edge_map ≈ 0) has weight 1×, while a pixel
    exactly on a colour boundary (edge_map = 1) has weight 1 + 3 = 4×.
    This 4× penalty for mis-reconstructed edge pixels forces the genetic
    algorithm to prioritise triangle placement along colour boundaries —
    the rings — before refining the flat-colour interior regions.

    The denominator Σ w × 3 normalises the result so it remains on a
    comparable scale to unweighted MSE and does not explode as image size grows.

    Parameters
    ----------
    rendered      : (H, W, 3) uint8
    target_f32    : (H, W, 3) float32
    edge_weights  : (H, W)    float32 in [0, 1], produced by compute_edge_map()

    Returns
    -------
    float  weighted MSE; lower is better; comparable scale to plain MSE.
    """
    diff_sq = (rendered.astype(np.float32) - target_f32) ** 2  # (H, W, 3)

    # Expand edge weights from (H, W) to (H, W, 1) so NumPy broadcasts the
    # same spatial weight across all three colour channels simultaneously.
    w = (1.0 + 3.0 * edge_weights)[:, :, np.newaxis]           # (H, W, 1)

    # Normalised weighted sum — dividing by Σ w × 3 keeps the output in the
    # same range as plain MSE, making the fitness history chart interpretable.
    return float(np.sum(diff_sq * w) / (np.sum(w) * 3.0))
