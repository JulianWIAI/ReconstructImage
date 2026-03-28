"""
dna_structures.py
Defines the core genetic data structures: Triangle and Genome.

Performance design:
  - Genome stores all triangle data as a single flat NumPy matrix
    of shape (N, 10): [x0, y0, x1, y1, x2, y2, r, g, b, alpha]
  - mutate() and crossover() operate on the full matrix in ONE
    vectorized NumPy call — zero Python for-loops in the hot path.
  - Triangle objects are constructed lazily (display only).

All internal state uses strict single-underscore private attributes.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional


# ---------------------------------------------------------------------------
# DNA column layout  (indices into each row of the (N, 10) matrix)
# ---------------------------------------------------------------------------
#   0  1   x0 y0  — vertex 0
#   2  3   x1 y1  — vertex 1
#   4  5   x2 y2  — vertex 2
#   6  7  8  r g b  — colour channels [0, 255]
#   9       alpha    — opacity [0.05, 1.0]

_COL_X = np.array([0, 2, 4], dtype=np.intp)   # x-coordinate columns
_COL_Y = np.array([1, 3, 5], dtype=np.intp)   # y-coordinate columns
_COL_RGB = np.array([6, 7, 8], dtype=np.intp) # colour columns
_COL_A = 9                                     # alpha column

# Per-column perturbation scale (used in Gaussian mutation)
# Will be multiplied by step_size and (w or h) where appropriate
_DELTA_SCALE_TEMPLATE = np.array(
    [0.15, 0.15,   # x0, y0  — fraction of image dimension
     0.15, 0.15,   # x1, y1
     0.15, 0.15,   # x2, y2
     40.0, 40.0, 40.0,  # r, g, b  — raw channel delta
     0.15],        # alpha
    dtype=np.float32,
)


class Triangle:
    """
    A single semi-transparent triangle gene.
    Used for display / introspection only; the hot rendering path
    reads directly from the Genome's DNA matrix.
    """

    __slots__ = ("_verts", "_rgba")

    def __init__(self, verts: np.ndarray, rgba: np.ndarray):
        # verts: (3, 2) float32  — [[x0,y0],[x1,y1],[x2,y2]]
        # rgba:  (4,)  float32  — [r, g, b, alpha]
        self._verts: np.ndarray = verts.astype(np.float32)
        self._rgba: np.ndarray = rgba.astype(np.float32)

    @property
    def vertices(self) -> np.ndarray:
        return self._verts

    @property
    def color(self) -> np.ndarray:
        return self._rgba


class Genome:
    """
    One candidate solution: N semi-transparent triangles encoded as a
    contiguous (N, 10) float32 NumPy array (_dna).

    Genetic operations (mutate / crossover) are fully vectorized —
    they operate on _dna with NumPy broadcasting, not Python loops.
    """

    def __init__(
        self,
        num_triangles: int,
        img_width: int = 128,
        img_height: int = 128,
        dna_matrix: Optional[np.ndarray] = None,
    ):
        self._num_triangles: int = num_triangles
        self._img_width: int = img_width
        self._img_height: int = img_height
        self._fitness: float = float("inf")

        if dna_matrix is not None:
            self._dna: np.ndarray = dna_matrix.astype(np.float32)
        else:
            self._dna = self._random_dna()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _random_dna(self) -> np.ndarray:
        """Create a random (N, 10) DNA matrix."""
        n = self._num_triangles
        w, h = self._img_width, self._img_height
        dna = np.empty((n, 10), dtype=np.float32)

        # Vertex x-coordinates  (columns 0, 2, 4)
        dna[:, _COL_X] = np.random.randint(0, w, (n, 3)).astype(np.float32)
        # Vertex y-coordinates  (columns 1, 3, 5)
        dna[:, _COL_Y] = np.random.randint(0, h, (n, 3)).astype(np.float32)
        # RGB channels          (columns 6, 7, 8)
        dna[:, _COL_RGB] = np.random.randint(0, 256, (n, 3)).astype(np.float32)
        # Alpha                 (column 9)
        dna[:, _COL_A] = np.random.uniform(0.10, 0.90, n).astype(np.float32)

        return dna

    def _clip_dna(self, dna: np.ndarray) -> np.ndarray:
        """Enforce valid ranges on every column (in-place, returns same array)."""
        w, h = self._img_width, self._img_height
        dna[:, _COL_X] = np.clip(dna[:, _COL_X], 0.0, w - 1.0)
        dna[:, _COL_Y] = np.clip(dna[:, _COL_Y], 0.0, h - 1.0)
        dna[:, _COL_RGB] = np.clip(dna[:, _COL_RGB], 0.0, 255.0)
        dna[:, _COL_A] = np.clip(dna[:, _COL_A], 0.05, 1.0)
        return dna

    # ------------------------------------------------------------------
    # Public read-only accessors
    # ------------------------------------------------------------------

    @property
    def dna(self) -> np.ndarray:
        """Raw (N, 10) float32 matrix — the canonical genome state."""
        return self._dna

    @property
    def fitness(self) -> float:
        return self._fitness

    @fitness.setter
    def fitness(self, value: float) -> None:
        self._fitness = float(value)

    @property
    def num_triangles(self) -> int:
        return self._num_triangles

    @property
    def img_width(self) -> int:
        return self._img_width

    @property
    def img_height(self) -> int:
        return self._img_height

    @property
    def triangles(self) -> List[Triangle]:
        """
        Lazily construct Triangle objects from the DNA matrix.
        Called only for display/introspection, not in the hot loop.
        """
        result: List[Triangle] = []
        for row in self._dna:
            verts = row[:6].reshape(3, 2)
            rgba = row[6:10]
            result.append(Triangle(verts, rgba))
        return result

    # ------------------------------------------------------------------
    # Vectorized genetic operations
    # ------------------------------------------------------------------

    def mutate(self, mutation_rate: float, step_size: float = 1.0) -> "Genome":
        """
        Vectorized mutation — ONE NumPy call for all N × 10 genes.

        step_size > 1.0 produces 'catastrophic' large jumps used for
        stagnation breaking.

        Returns a new Genome; does NOT modify self.
        """
        w, h = self._img_width, self._img_height
        n = self._num_triangles

        # Build per-column absolute delta scales
        scale = _DELTA_SCALE_TEMPLATE.copy()
        scale[_COL_X] *= w  # convert fraction → pixels
        scale[_COL_Y] *= h
        scale *= step_size

        # Gaussian perturbations shaped (N, 10), scaled per column
        deltas = np.random.randn(n, 10).astype(np.float32) * scale[np.newaxis, :]

        # Bernoulli gate: each gene mutates independently
        gate = (np.random.random((n, 10)) < mutation_rate).astype(np.float32)

        new_dna = self._dna + deltas * gate
        self._clip_dna(new_dna)

        child = Genome(n, w, h, dna_matrix=new_dna)
        return child

    def crossover(self, other: "Genome") -> "Genome":
        """
        Uniform crossover: for each triangle, randomly inherit from
        either parent. Vectorized via boolean mask + np.where.
        """
        # mask shape (N, 1) → broadcasts to (N, 10)
        mask = (np.random.random(self._num_triangles) < 0.5)[:, np.newaxis]
        child_dna = np.where(mask, self._dna, other._dna)
        return Genome(self._num_triangles, self._img_width, self._img_height,
                      dna_matrix=child_dna)

    def clone(self) -> "Genome":
        child = Genome(
            self._num_triangles,
            self._img_width,
            self._img_height,
            dna_matrix=self._dna.copy(),
        )
        child._fitness = self._fitness
        return child
