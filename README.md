# Evolutionary Image Reconstruction
### A Genetic Algorithm Showcase — JOSEPHS Exhibition

![Showcase](placeholder_image.png)

---

## Project Overview

This project uses a fully custom **Genetic Algorithm (GA)** — a bio-inspired optimisation technique modelled on natural selection — to reconstruct a target image using nothing but semi-transparent, overlapping triangles. Each candidate solution (a *genome*) is a collection of triangles with random positions and colours; over thousands of generations the population *evolves*, with the fittest individuals surviving, reproducing, and gradually converging on the target. The result is a living demonstration of how evolutionary computation can solve visual problems without any gradient information or neural networks.

---

## Key Features

- **Fully custom Genetic Algorithm** — no external GA library; every operator is hand-crafted in pure Python/NumPy
- **Vectorised NumPy renderer** — barycentric triangle rasterisation with Porter-Duff alpha blending; zero per-triangle Pillow allocations in the hot path
- **Edge-Weighted MSE fitness function** — a custom Sobel-based edge map amplifies the fitness penalty at colour boundaries by up to 4×, forcing the algorithm to reconstruct sharp ring edges before flat interior regions
- **Tournament selection** — rank-based parent picking (k = 4); scale-invariant and robust across all fitness magnitudes
- **Elitism** — the top 10 % of each generation survive unchanged, guaranteeing monotonically non-decreasing best-fitness
- **Stagnation detection & catastrophic mutation burst** — if the population's best MSE does not improve by more than 0.5 units over a configurable window, the algorithm triggers a 3× mutation amplification burst and regenerates 60 % of the population to escape local optima
- **Edge-biased vertex initialisation** — on startup and after each burst, a fraction of genome vertices are snapped to high-gradient pixels, injecting domain knowledge about where important features are located
- **Adaptive mutation rate** — the burst rate decays exponentially back to the user-set baseline (~14 generations to halve), preventing runaway exploration after recovery
- **Dual-resolution strategy** — fitness is computed on a 128 × 128 downsampled image for speed; the dashboard renders the original full-resolution image for visual quality
- **Real-time Streamlit dashboard** — live fitness chart, generation counter, gens/sec throughput, burst indicator, and side-by-side target vs. candidate comparison; all placeholders populated in a single render pass to eliminate sequential load lag

---

## Tech Stack

| Layer | Library |
|---|---|
| UI & State Management | [Streamlit](https://streamlit.io/) ≥ 1.32 |
| Numerical Computing & Rendering | [NumPy](https://numpy.org/) ≥ 1.24 |
| Image I/O & Display | [Pillow](https://python-pillow.org/) ≥ 10.0 |
| Fitness Chart Data | [Pandas](https://pandas.pydata.org/) ≥ 2.0 |
| Language | Python 3.11+ |

> **No external GA framework is used.** The entire evolutionary engine — selection, crossover, mutation, elitism, and stagnation handling — is implemented from scratch.

---

## Project Structure

```
ReconstructImage/
├── main.py                      # Application entry point
├── requirements.txt
├── assets/                      # All static resources
│   ├── icon.png                 # Browser tab favicon
│   ├── BatmanLogoTest.png
│   ├── JosephsLogo.png
│   └── generate_assets.py
└── SBS/                         # Core package
    ├── __init__.py
    ├── dna_structures.py        # Genome & Triangle data structures
    ├── evolutionary_engine.py   # GA loop: selection, crossover, elitism
    ├── image_utils.py           # Renderer, Sobel edge map, fitness functions
    └── dashboard_controller.py  # Streamlit UI & session-state orchestration
```

---

## Installation & Execution

**Prerequisites:** Python 3.11 or higher, `pip`, and `git`.

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ReconstructImage.git
cd ReconstructImage
```

### 2. Create and activate a virtual environment *(recommended)*

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run main.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`.

### 5. Usage

1. Select a **target image** from the sidebar dropdown.
2. Adjust the **hyperparameters** (mutation rate, triangle count, population size, stagnation threshold).
3. Click **Start** — watch the genetic algorithm reconstruct the image in real time.
4. Click **Stop** at any time to pause, then **Start** again to resume with new settings.

---

## How It Works

```
Initialise population (edge-biased random genomes)
        │
        ▼
┌─── Evaluate fitness (Edge-Weighted MSE for every genome) ◄──────────┐
│       │                                                               │
│       ▼                                                               │
│   Stagnation check ──► Catastrophic burst (3× mutation + re-inject) ─┤
│       │                                                               │
│       ▼                                                               │
│   Tournament selection (k = 4)                                        │
│       │                                                               │
│       ▼                                                               │
│   Uniform crossover + Gaussian mutation                               │
│       │                                                               │
│   Elitism: top 10 % survive unchanged                                 │
│       │                                                               │
└───────┴── Next generation ────────────────────────────────────────────┘
```

The fitness landscape is guided by a **Sobel-computed edge map** pre-calculated once from the target image. Pixels on colour boundaries carry up to **4× the fitness weight** of flat interior pixels, directly shaping where the algorithm places triangles.

---

## Development Process

This scientific showcase was architected and developed for the JOSEPHS Exhibition. The underlying simulation logic, genetic algorithm tuning, and UI rendering were co-developed with Artificial Intelligence (Claude 3.5 Sonnet / Gemini) to demonstrate modern AI-assisted software engineering workflows.

---

## License

This project is licensed under the [MIT License](LICENSE).