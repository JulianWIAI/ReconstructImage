"""
evolutionary_engine.py
Core evolutionary algorithm logic with stagnation detection and
edge-guided optimisation.

Class hierarchy:
    BaseEvolutionHandler     (ABC — defines the EA contract)
        └── ImageReconstructor  (concrete — reconstructs images from triangles)

Performance and optimisation features:
  - Vectorised evaluation using the fast NumPy renderer.
  - Tournament selection with fully vectorised index sampling.
  - Uniform crossover on the (N, 10) DNA matrix (single np.where call).
  - Stagnation detector: if best fitness does not improve by at least
    _STAGNATION_TOLERANCE for _stagnation_threshold consecutive generations,
    a 'catastrophic mutation' burst is triggered to escape local optima.
  - Edge-biased vertex injection: on stagnation, newly regenerated genomes
    have vertices sampled near the image's colour-boundary pixels.
  - Adaptive mutation rate: rises 3× during a stagnation burst, then
    decays back to the user-set base rate exponentially.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from SBS.dna_structures import Genome
from SBS.image_utils import (
    render_genome_fast,
    compute_fitness_weighted,
    get_high_edge_coords,
    compute_edge_map,
)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseEvolutionHandler(ABC):
    """
    Defines the public interface and shared bookkeeping for any
    population-based evolutionary optimiser.

    Subclasses must implement:
        _create_initial_population()
        _evaluate_population()
        _select_parents()
        _breed_next_generation()
    """

    def __init__(self, population_size: int, mutation_rate: float):
        self._population_size: int = population_size
        self._mutation_rate: float = mutation_rate
        self._population: List[Genome] = []
        self._current_generation: int = 0
        self._best_genome: Optional[Genome] = None
        self._fitness_history: List[float] = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_initial_population(self) -> List[Genome]:
        """Return a freshly randomised population."""

    @abstractmethod
    def _evaluate_population(self) -> None:
        """Assign a fitness score to every genome in the population."""

    @abstractmethod
    def _select_parents(self) -> List[Genome]:
        """Select genomes that will reproduce."""

    @abstractmethod
    def _breed_next_generation(self, parents: List[Genome]) -> List[Genome]:
        """Produce the next generation from selected parents."""

    # ------------------------------------------------------------------
    # Shared public API
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Seed the population. Must be called before step()."""
        self._population = self._create_initial_population()
        self._current_generation = 0
        self._fitness_history = []
        self._best_genome = None

    def step(self) -> Tuple[Genome, float]:
        """
        Advance the algorithm by one generation.
        Returns (best_genome, best_fitness).
        """
        self._evaluate_population()
        self._update_best()
        parents = self._select_parents()
        self._population = self._breed_next_generation(parents)
        self._current_generation += 1
        return self._best_genome, self._best_genome.fitness

    def _update_best(self) -> None:
        """Track the elite genome across generations."""
        current_best = min(self._population, key=lambda g: g.fitness)
        if self._best_genome is None or current_best.fitness < self._best_genome.fitness:
            self._best_genome = current_best.clone()
        self._fitness_history.append(self._best_genome.fitness)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def current_generation(self) -> int:
        return self._current_generation

    @property
    def best_genome(self) -> Optional[Genome]:
        return self._best_genome

    @property
    def fitness_history(self) -> List[float]:
        return list(self._fitness_history)

    @property
    def population_size(self) -> int:
        return self._population_size

    @property
    def mutation_rate(self) -> float:
        return self._mutation_rate


# ---------------------------------------------------------------------------
# Concrete image reconstructor
# ---------------------------------------------------------------------------

class ImageReconstructor(BaseEvolutionHandler):
    """
    Evolutionary algorithm that reconstructs a target image using a fixed
    number of semi-transparent triangles.

    Optimisation strategy
    ---------------------
    Selection   : Tournament (k = _TOURNAMENT_SIZE)
    Crossover   : Uniform crossover on the flat DNA matrix (np.where)
    Elitism     : Top _ELITE_FRACTION survive unchanged
    Fitness     : Edge-weighted MSE (boundary pixels matter more)
    Stagnation  : Catastrophic mutation burst + edge-biased re-injection
    Adaptation  : Mutation rate decays exponentially after each burst
    """

    # Class-level constants
    _ELITE_FRACTION: float = 0.10      # fraction of population preserved unchanged
    _TOURNAMENT_SIZE: int = 4          # contestants per tournament
    _STAGNATION_TOLERANCE: float = 0.5 # min MSE improvement to NOT be stagnant
    _MUTATION_BURST_FACTOR: float = 3.0  # how many times to amplify rate on burst
    _MUTATION_DECAY: float = 0.95      # exponential decay after burst
    _EDGE_INJECT_FRACTION: float = 0.4 # fraction of post-stagnation genomes
                                        # that get edge-biased vertices

    def __init__(
        self,
        target_image: np.ndarray,
        num_triangles: int,
        population_size: int,
        mutation_rate: float,
        stagnation_threshold: int = 60,
    ):
        super().__init__(
            population_size=population_size,
            mutation_rate=mutation_rate,
        )
        self._target_image: np.ndarray = target_image
        self._target_f32: np.ndarray = target_image.astype(np.float32)
        self._num_triangles: int = num_triangles
        self._img_height: int
        self._img_width: int
        self._img_height, self._img_width = target_image.shape[:2]
        self._elite_count: int = max(1, int(population_size * self._ELITE_FRACTION))

        # Stagnation state
        self._stagnation_threshold: int = stagnation_threshold
        self._stagnation_counter: int = 0
        self._base_mutation_rate: float = mutation_rate
        self._current_mutation_rate: float = mutation_rate
        self._in_burst: bool = False

        # Pre-compute edge map once (pure NumPy Sobel)
        self._edge_map: np.ndarray = compute_edge_map(target_image)
        self._high_edge_coords: np.ndarray = get_high_edge_coords(
            self._edge_map, threshold=0.25
        )

    # ------------------------------------------------------------------
    # BaseEvolutionHandler implementations
    # ------------------------------------------------------------------

    def _create_initial_population(self) -> List[Genome]:
        """
        Seed population with edge-biased vertex placement.
        Roughly half of each genome's triangles have at least one vertex
        sampled near a high-gradient pixel.
        """
        pop = []
        w, h = self._img_width, self._img_height
        n = self._num_triangles
        for _ in range(self._population_size):
            g = Genome(n, w, h)
            g = self._inject_edge_vertices(g, fraction=0.5)
            pop.append(g)
        return pop

    def _evaluate_population(self) -> None:
        """
        Render every genome and compute edge-weighted MSE.
        The renderer is pure NumPy (no Pillow allocation per triangle).
        """
        w, h = self._img_width, self._img_height
        for genome in self._population:
            rendered = render_genome_fast(genome.dna, w, h)
            genome.fitness = compute_fitness_weighted(
                rendered, self._target_f32, self._edge_map
            )

    def _select_parents(self) -> List[Genome]:
        """
        Vectorised tournament selection.
        All tournament indices are sampled in a single np.random.choice call,
        then the per-tournament minimum is found with a fast loop over k slices.
        """
        pop = self._population
        n = len(pop)
        k = self._TOURNAMENT_SIZE
        num_parents = self._population_size

        # Sample all contestants at once: (num_parents, k)
        all_contestants = np.random.randint(0, n, size=(num_parents, k))

        parents: List[Genome] = []
        fitnesses = np.array([g.fitness for g in pop], dtype=np.float64)

        for row in all_contestants:
            # row is a 1-D array of k indices
            winner_idx = row[np.argmin(fitnesses[row])]
            parents.append(pop[winner_idx])

        return parents

    def _breed_next_generation(self, parents: List[Genome]) -> List[Genome]:
        """
        Elitism + uniform crossover + vectorised mutation.
        Elites are cloned unchanged; offspring are bred from tournament winners.
        """
        # Elites survive unchanged
        sorted_pop = sorted(self._population, key=lambda g: g.fitness)
        elites = [g.clone() for g in sorted_pop[: self._elite_count]]

        offspring: List[Genome] = []
        remaining = self._population_size - self._elite_count
        n_parents = len(parents)

        # Batch-sample parent pairs
        pairs = np.random.randint(0, n_parents, size=(remaining, 2))

        rate = self._current_mutation_rate
        for a_idx, b_idx in pairs:
            while b_idx == a_idx:
                b_idx = np.random.randint(0, n_parents)
            child = parents[a_idx].crossover(parents[b_idx])
            child = child.mutate(mutation_rate=rate)
            offspring.append(child)

        return elites + offspring

    # ------------------------------------------------------------------
    # Stagnation detection and catastrophic mutation
    # ------------------------------------------------------------------

    def _check_and_handle_stagnation(self) -> bool:
        """
        Called after _update_best().
        Returns True if a stagnation burst was triggered this generation.
        """
        hist = self._fitness_history
        if len(hist) < self._stagnation_threshold:
            return False

        recent_window = hist[-self._stagnation_threshold:]
        improvement = recent_window[0] - recent_window[-1]

        if improvement < self._STAGNATION_TOLERANCE:
            self._trigger_catastrophic_mutation()
            return True

        # Decay the mutation rate back toward base if not stagnating
        if self._in_burst:
            self._current_mutation_rate = max(
                self._base_mutation_rate,
                self._current_mutation_rate * self._MUTATION_DECAY,
            )
            if abs(self._current_mutation_rate - self._base_mutation_rate) < 1e-4:
                self._in_burst = False

        return False

    def _trigger_catastrophic_mutation(self) -> None:
        """
        Escape a local optimum by:
          1. Amplifying the mutation rate 3×.
          2. Regenerating the bottom 60 % of the population with large
             step-size mutations of the current elites.
          3. Injecting edge-biased vertices into a fraction of the
             regenerated genomes so they immediately orient toward
             colour boundaries (the JOSEPHS rings).
        """
        self._current_mutation_rate = min(
            0.8,
            self._base_mutation_rate * self._MUTATION_BURST_FACTOR,
        )
        self._in_burst = True
        self._stagnation_counter = 0

        n_keep = max(2, int(self._population_size * 0.40))
        sorted_pop = sorted(self._population, key=lambda g: g.fitness)
        survivors = [g.clone() for g in sorted_pop[:n_keep]]

        n_inject = max(1, int(
            (self._population_size - n_keep) * self._EDGE_INJECT_FRACTION
        ))
        n_random = (self._population_size - n_keep) - n_inject

        regenerated: List[Genome] = []

        # Edge-biased injections: mutate survivors heavily + snap to edges
        for i in range(n_inject):
            parent = survivors[i % n_keep]
            child = parent.mutate(mutation_rate=self._current_mutation_rate,
                                  step_size=2.5)
            child = self._inject_edge_vertices(child, fraction=0.5)
            regenerated.append(child)

        # Purely random large-step mutations
        for i in range(n_random):
            parent = survivors[i % n_keep]
            child = parent.mutate(mutation_rate=self._current_mutation_rate,
                                  step_size=3.0)
            regenerated.append(child)

        self._population = survivors + regenerated

    # ------------------------------------------------------------------
    # Edge-guided vertex injection
    # ------------------------------------------------------------------

    def _inject_edge_vertices(self, genome: Genome, fraction: float) -> Genome:
        """
        Replace `fraction` of the genome's triangle vertices with positions
        sampled near high-gradient (edge) pixels.
        Returns a new Genome; original is not modified.
        """
        if len(self._high_edge_coords) == 0:
            return genome

        dna = genome.dna.copy()
        n = genome.num_triangles
        num_to_snap = max(1, int(n * fraction))

        # Choose which triangles to modify
        tri_indices = np.random.choice(n, size=num_to_snap, replace=False)
        # Sample edge coords
        edge_sample_indices = np.random.randint(
            0, len(self._high_edge_coords), size=num_to_snap
        )
        sampled = self._high_edge_coords[edge_sample_indices]  # (K, 2): [row, col]

        w, h = self._img_width, self._img_height

        for k, tri_i in enumerate(tri_indices):
            # For each chosen triangle, snap one random vertex to an edge pixel
            v = np.random.randint(0, 3)  # vertex index 0, 1, or 2
            row, col = sampled[k]

            # Add small Gaussian jitter so they don't pile on the same pixel
            jitter_x = np.random.randint(-4, 5)
            jitter_y = np.random.randint(-4, 5)

            dna[tri_i, v * 2]     = float(np.clip(col + jitter_x, 0, w - 1))
            dna[tri_i, v * 2 + 1] = float(np.clip(row + jitter_y, 0, h - 1))

        return Genome(
            genome.num_triangles,
            genome.img_width,
            genome.img_height,
            dna_matrix=dna,
        )

    # ------------------------------------------------------------------
    # Overridden step() — adds stagnation check
    # ------------------------------------------------------------------

    def step(self) -> Tuple[Genome, float]:
        """
        One full generation: evaluate → track best → stagnation check →
        select → breed → advance counter.
        """
        self._evaluate_population()
        self._update_best()
        self._check_and_handle_stagnation()
        parents = self._select_parents()
        self._population = self._breed_next_generation(parents)
        self._current_generation += 1
        return self._best_genome, self._best_genome.fitness

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def num_triangles(self) -> int:
        return self._num_triangles

    @property
    def target_image(self) -> np.ndarray:
        return self._target_image

    @property
    def is_in_burst(self) -> bool:
        """True when a catastrophic mutation burst is currently active."""
        return self._in_burst

    @property
    def current_mutation_rate(self) -> float:
        return self._current_mutation_rate

    @property
    def stagnation_counter(self) -> int:
        return self._stagnation_counter

    # ------------------------------------------------------------------
    # Re-initialisation helper (called from main.py on settings change)
    # ------------------------------------------------------------------

    def update_hyperparameters(
        self,
        mutation_rate: float,
        num_triangles: int,
        population_size: int,
    ) -> None:
        """Rebuild optimizer with new hyperparameters and reset state."""
        self._base_mutation_rate = mutation_rate
        self._current_mutation_rate = mutation_rate
        self._mutation_rate = mutation_rate
        self._num_triangles = num_triangles
        self._population_size = population_size
        self._elite_count = max(1, int(population_size * self._ELITE_FRACTION))
        self._img_height, self._img_width = self._target_image.shape[:2]
        self._in_burst = False
        self._stagnation_counter = 0
        self.initialize()
