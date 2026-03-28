"""
SBS/dashboard_controller.py
Encapsulates the entire Streamlit UI, state management, and EA execution
loop for the JOSEPHS Evolutionary Image Reconstruction exhibit.

Public interface
----------------
    from SBS.dashboard_controller import ShowcaseDashboard
    ShowcaseDashboard().run()

Design rules
------------
  - All internal state uses single-underscore private attributes.
  - Business logic (EA, rendering, fitness) lives in the other SBS modules.
  - This class only orchestrates the UI; it never touches NumPy directly.
  - The three startup bugs are fixed here:
      1. App no longer auto-starts on page load.
      2. Changing the target image stops the run but does NOT restart it.
      3. The full layout (target image, chart skeleton, candidate placeholder)
         is rendered in one pass on every rerun — no sequential load lag.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from SBS.evolutionary_engine import ImageReconstructor
from SBS.image_utils import (
    genome_to_pil,
    list_asset_images,
    load_and_resize,
    load_full_res,
)


class ShowcaseDashboard:
    """
    Streamlit dashboard for the Evolutionary Image Reconstruction exhibit.

    Lifecycle per Streamlit rerun
    ------------------------------
    __init__()          — page config (must be first Streamlit call)
    run()
      _apply_styles()   — inject CSS
      _init_session()   — seed session_state + pre-load default target image
      _render_sidebar() — draw controls, store widget values as instance vars
      _handle_controls()— react to button/dropdown events (no auto-start)
      _build_layout()   — create all st.empty() placeholders in one pass
      _refresh_display()— fill every placeholder immediately (eliminates lag)
      _run_evolution_loop() — blocked-while loop; only entered when running
    """

    # ------------------------------------------------------------------
    # Private class-level constants
    # ------------------------------------------------------------------
    _ASSETS_DIR: Path = Path(__file__).parent.parent / "assets"
    _FITNESS_RENDER_SIZE: int = 128   # low-res used for MSE (biggest speed win)
    _MAX_GENERATIONS: int = 10_000
    _STEPS_PER_FRAME: int = 4         # EA steps between Streamlit refreshes
    _CHART_UPDATE_EVERY: int = 20     # throttle chart redraws
    _DEFAULT_STAGNATION: int = 60
    _IMG_DISPLAY_WIDTH: int = 280     # fixed px width keeps chart visible

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """
        Configure the Streamlit page.
        st.set_page_config() MUST be the first Streamlit call in the
        script; placing it here guarantees that before run() does anything.
        """
        st.set_page_config(
            page_title="JOSEPHS | Evolutionary Image Reconstruction",
            page_icon="DNA",
            layout="wide",
        )

        # Sidebar widget values — populated by _render_sidebar()
        self._selected_image: str = ""
        self._mutation_rate: float = 0.04
        self._num_triangles: int = 60
        self._population_size: int = 25
        self._stagnation_threshold: int = self._DEFAULT_STAGNATION
        self._start_btn: bool = False
        self._stop_btn: bool = False

        # Layout placeholders — populated by _build_layout()
        self._ph_gen = None
        self._ph_mse = None
        self._ph_gen_s = None
        self._ph_triangles = None
        self._ph_burst = None
        self._ph_target = None
        self._ph_candidate = None
        self._ph_chart = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Drive a single Streamlit rerun from top to bottom."""
        self._apply_styles()
        self._init_session()
        self._render_sidebar()
        self._handle_controls()
        self._build_layout()
        self._refresh_display()
        self._run_evolution_loop()

    # ------------------------------------------------------------------
    # Private: one-time / session setup
    # ------------------------------------------------------------------

    def _apply_styles(self) -> None:
        """Inject minimal CSS for the showroom dark-sidebar aesthetic."""
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] { background: #0f0f1a; }
            [data-testid="stSidebar"] * { color: #e0e0e0; }
            .block-container { padding-top: 1.5rem; }
            h1 { letter-spacing: -0.5px; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def _init_session(self) -> None:
        """
        Seed session_state with defaults on the very first visit.

        Bug-3 fix: the default target image is pre-loaded here so that
        _refresh_display() can populate the target image placeholder on
        the first render — before the user presses Start.  This eliminates
        the sequential load lag where the target appeared seconds after
        the page loaded.
        """
        available = list_asset_images(str(self._ASSETS_DIR))
        default_image = available[0] if available else ""

        defaults: dict = {
            "optimizer": None,
            "running": False,
            "generation": 0,
            "fitness_history": [],
            "gen_timestamps": [],
            "best_image": None,
            "target_array_small": None,    # 128×128 for fitness
            "target_array_display": None,  # high-res for display
            "selected_image": default_image,
            "hp_mutation_rate": 0.04,
            "hp_num_triangles": 60,
            "hp_population_size": 25,
            "hp_stagnation": self._DEFAULT_STAGNATION,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

        # Pre-load the default target once (subsequent reruns skip this)
        if st.session_state["target_array_small"] is None and default_image:
            self._load_target_images(default_image)

    def _load_target_images(self, image_name: str) -> None:
        """
        Load both resolutions of a target image into session_state.
        Low-res (128×128) is used for fitness; high-res for display only.
        """
        if not image_name:
            return
        path = str(self._ASSETS_DIR / image_name)
        st.session_state["target_array_small"] = load_and_resize(
            path, self._FITNESS_RENDER_SIZE
        )
        st.session_state["target_array_display"] = load_full_res(
            path, max_display_size=400
        )

    # ------------------------------------------------------------------
    # Private: sidebar rendering
    # ------------------------------------------------------------------

    def _render_sidebar(self) -> None:
        """
        Draw all sidebar controls and store their current values as
        private instance variables for use in _handle_controls().
        """
        with st.sidebar:
            st.title("JOSEPHS EA Demo")
            st.caption("Evolutionary Image Reconstruction")
            st.divider()
            st.header("Target Image")

            available = list_asset_images(str(self._ASSETS_DIR))
            if not available:
                st.error(
                    "No images found in /assets/. "
                    "Run `python assets/generate_assets.py` first."
                )
                st.stop()

            # Restore the previously selected image so reruns don't reset
            saved_image = st.session_state.get("selected_image", available[0])
            default_idx = (
                available.index(saved_image) if saved_image in available else 0
            )
            self._selected_image = st.selectbox(
                "Select target",
                options=available,
                index=default_idx,
                help="The image the genetic algorithm will try to reconstruct.",
            )

            st.divider()
            st.header("Hyperparameters")

            self._mutation_rate = st.slider(
                "Mutation Rate",
                min_value=0.001,
                max_value=0.40,
                value=st.session_state["hp_mutation_rate"],
                step=0.001,
                format="%.3f",
                help=(
                    "Probability each gene is perturbed per generation. "
                    "Higher = more exploration; lower = more refinement."
                ),
            )
            self._num_triangles = st.slider(
                "Number of Triangles",
                min_value=10,
                max_value=200,
                value=st.session_state["hp_num_triangles"],
                step=5,
                help="Genome length. More triangles = higher fidelity but slower.",
            )
            self._population_size = st.slider(
                "Population Size",
                min_value=5,
                max_value=80,
                value=st.session_state["hp_population_size"],
                step=5,
                help="Candidate solutions evaluated each generation.",
            )
            self._stagnation_threshold = st.slider(
                "Stagnation Threshold (gens)",
                min_value=10,
                max_value=200,
                value=st.session_state["hp_stagnation"],
                step=10,
                help=(
                    "Generations without MSE improvement before a "
                    "catastrophic mutation burst fires to escape local optima."
                ),
            )

            st.divider()
            col_start, col_stop = st.columns(2)
            self._start_btn = col_start.button(
                "Start", use_container_width=True, type="primary"
            )
            self._stop_btn = col_stop.button(
                "Stop", use_container_width=True
            )

            st.divider()
            st.markdown("**Algorithm details**")
            st.markdown(
                "- Selection: Tournament (k=4)\n"
                "- Crossover: Uniform per-triangle\n"
                "- Elitism: top 10% survive\n"
                "- Fitness: Edge-weighted MSE\n"
                "- Stagnation: Catastrophic burst\n"
                "- Init: Edge-biased vertices"
            )

    # ------------------------------------------------------------------
    # Private: control logic
    # ------------------------------------------------------------------

    def _handle_controls(self) -> None:
        """
        Translate widget events into session_state mutations.

        Bug-1 fix: The app no longer starts automatically on page load.
          session_state["running"] starts as False and only becomes True
          when the user explicitly clicks Start.

        Bug-2 fix: Changing the target image (or any dropdown value) stops
          a running session and reloads images, but does NOT set running=True.
          The user must press Start again.

        Only self._start_btn sets running=True.
        """
        # --- React to target image change ----------------------------------
        if self._selected_image != st.session_state["selected_image"]:
            st.session_state["selected_image"] = self._selected_image
            self._load_target_images(self._selected_image)
            # Stop any running session; do NOT auto-start
            st.session_state["running"] = False
            st.session_state["optimizer"] = None
            st.session_state["generation"] = 0
            st.session_state["fitness_history"] = []
            st.session_state["gen_timestamps"] = []
            st.session_state["best_image"] = None

        # --- Stop button ---------------------------------------------------
        if self._stop_btn:
            st.session_state["running"] = False

        # --- Start button (the ONLY place running becomes True) ------------
        if self._start_btn:
            # Persist the current slider values so they survive reruns
            st.session_state["hp_mutation_rate"] = self._mutation_rate
            st.session_state["hp_num_triangles"] = self._num_triangles
            st.session_state["hp_population_size"] = self._population_size
            st.session_state["hp_stagnation"] = self._stagnation_threshold

            # Ensure target images are available (defensive: covers edge cases
            # where session_state was cleared between reruns)
            if st.session_state["target_array_small"] is None:
                self._load_target_images(self._selected_image)

            # Build a fresh optimizer with the current settings
            optimizer = ImageReconstructor(
                target_image=st.session_state["target_array_small"],
                num_triangles=self._num_triangles,
                population_size=self._population_size,
                mutation_rate=self._mutation_rate,
                stagnation_threshold=self._stagnation_threshold,
            )
            optimizer.initialize()

            st.session_state["optimizer"] = optimizer
            st.session_state["running"] = True
            st.session_state["generation"] = 0
            st.session_state["fitness_history"] = []
            st.session_state["gen_timestamps"] = []
            st.session_state["best_image"] = None

    # ------------------------------------------------------------------
    # Private: layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """
        Construct the static page skeleton and create every st.empty()
        placeholder in a single top-to-bottom pass.

        All placeholders are assigned to instance variables so that both
        _refresh_display() and _run_evolution_loop() can update them by
        name without re-querying the DOM.
        """
        st.title("Evolutionary Image Reconstruction - JOSEPHS Exhibit")
        st.caption(
            "A genetic algorithm reconstructs the target image using "
            "semi-transparent triangles. Watch MSE descend as the "
            "population learns."
        )

        # Metric row — five equal columns
        c1, c2, c3, c4, c5 = st.columns(5)
        self._ph_gen = c1.empty()
        self._ph_mse = c2.empty()
        self._ph_gen_s = c3.empty()
        self._ph_triangles = c4.empty()
        self._ph_burst = c5.empty()

        st.divider()

        # Image comparison row
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.caption("Target")
            self._ph_target = st.empty()
        with img_col2:
            st.caption("Best Candidate (live)")
            self._ph_candidate = st.empty()

        # Fitness chart below images — always present in DOM
        st.caption(
            "Fitness (Edge-Weighted MSE) over Generations — lower is better"
        )
        self._ph_chart = st.empty()

    # ------------------------------------------------------------------
    # Private: initial display fill
    # ------------------------------------------------------------------

    def _refresh_display(self) -> None:
        """
        Populate every placeholder with the current session_state values.

        Bug-3 fix: this is called once per rerun, before the EA loop.
        It fills ALL placeholders immediately so the page appears complete
        in a single render — no elements loading seconds apart.

        The target image is available immediately because _init_session()
        pre-loads it.  The candidate and chart slots show informative
        placeholder messages until the EA produces actual data.
        """
        gen = st.session_state["generation"]
        history: list = st.session_state["fitness_history"]
        is_running: bool = st.session_state["running"]
        current_fit = history[-1] if history else None
        opt = st.session_state.get("optimizer")

        # --- Metrics -------------------------------------------------------
        self._ph_gen.metric("Generation", f"{gen:,}")
        self._ph_mse.metric(
            "Fitness (MSE)",
            f"{current_fit:.2f}" if current_fit is not None else "-",
        )
        self._ph_triangles.metric(
            "Triangles", st.session_state["hp_num_triangles"]
        )

        ts = st.session_state["gen_timestamps"]
        if len(ts) >= 2:
            rate = (len(ts) - 1) / max(ts[-1] - ts[0], 1e-9)
            self._ph_gen_s.metric("Gens / sec", f"{rate:.1f}")
        else:
            self._ph_gen_s.metric("Gens / sec", "-")

        self._ph_burst.metric(
            "Burst active", "YES" if (opt and opt.is_in_burst) else "no"
        )

        # --- Target image (always visible — pre-loaded on first visit) -----
        tgt = st.session_state["target_array_display"]
        if tgt is not None:
            self._ph_target.image(
                Image.fromarray(tgt), width=self._IMG_DISPLAY_WIDTH
            )
        else:
            self._ph_target.info("Select a target image from the sidebar.")

        # --- Best candidate (placeholder until EA produces output) ---------
        best = st.session_state["best_image"]
        if best is not None:
            self._ph_candidate.image(
                best.resize((256, 256), Image.Resampling.NEAREST),
                width=self._IMG_DISPLAY_WIDTH,
            )
        elif is_running:
            self._ph_candidate.info("Generating first candidate...")
        else:
            self._ph_candidate.info(
                "Press **Start** to begin reconstruction."
            )

        # --- Fitness chart (placeholder until enough data points) ----------
        if len(history) >= 2:
            self._ph_chart.line_chart(
                pd.DataFrame({"MSE": history}), height=180
            )
        elif is_running:
            self._ph_chart.info("Collecting first data points...")
        else:
            self._ph_chart.info(
                "Press **Start** — the fitness chart will appear here."
            )

    # ------------------------------------------------------------------
    # Private: EA execution loop
    # ------------------------------------------------------------------

    def _run_evolution_loop(self) -> None:
        """
        Run the evolutionary algorithm until stopped or MAX_GENERATIONS
        is reached.  Advances _STEPS_PER_FRAME generations between each
        Streamlit UI update to maximise throughput while keeping the
        display responsive.

        This method is a no-op when the algorithm is not running, so
        Streamlit can complete its normal render cycle undisturbed.
        """
        if not st.session_state["running"] or st.session_state["optimizer"] is None:
            return

        optimizer: ImageReconstructor = st.session_state["optimizer"]
        generation: int = st.session_state["generation"]
        fitness_history: list = st.session_state["fitness_history"]
        gen_timestamps: list = st.session_state["gen_timestamps"]

        while st.session_state["running"] and generation < self._MAX_GENERATIONS:

            # ---- Advance the EA by STEPS_PER_FRAME generations -----------
            for _ in range(self._STEPS_PER_FRAME):
                best_genome, best_fitness = optimizer.step()
                generation += 1
                fitness_history.append(best_fitness)
                gen_timestamps.append(time.perf_counter())

            # Persist state back to session_state
            best_pil = genome_to_pil(best_genome)
            st.session_state["generation"] = generation
            st.session_state["fitness_history"] = fitness_history
            st.session_state["gen_timestamps"] = gen_timestamps
            st.session_state["best_image"] = best_pil

            # ---- Update UI placeholders ----------------------------------

            # Metrics (every frame)
            self._ph_gen.metric("Generation", f"{generation:,}")
            self._ph_mse.metric("Fitness (MSE)", f"{best_fitness:.2f}")
            self._ph_triangles.metric(
                "Triangles", st.session_state["hp_num_triangles"]
            )
            self._ph_burst.metric(
                "Burst active", "YES" if optimizer.is_in_burst else "no"
            )

            ts_window = gen_timestamps[-50:]
            if len(ts_window) >= 2:
                rate = (len(ts_window) - 1) / max(
                    ts_window[-1] - ts_window[0], 1e-9
                )
                self._ph_gen_s.metric("Gens / sec", f"{rate:.1f}")

            # Candidate image (every frame)
            self._ph_candidate.image(
                best_pil.resize((256, 256), Image.Resampling.NEAREST),
                width=self._IMG_DISPLAY_WIDTH,
            )

            # Target image — only on first frame; it never changes mid-run
            if generation <= self._STEPS_PER_FRAME:
                tgt = st.session_state["target_array_display"]
                if tgt is not None:
                    self._ph_target.image(
                        Image.fromarray(tgt), width=self._IMG_DISPLAY_WIDTH
                    )

            # Fitness chart — throttled to reduce JSON serialisation overhead
            if (
                generation % self._CHART_UPDATE_EVERY == 0
                and len(fitness_history) >= 2
            ):
                self._ph_chart.line_chart(
                    pd.DataFrame({"MSE": fitness_history}), height=180
                )

            # Brief yield so Streamlit can process the Stop button
            time.sleep(0.001)

        # Natural end of run
        if generation >= self._MAX_GENERATIONS:
            st.session_state["running"] = False
            st.success(
                f"Completed {self._MAX_GENERATIONS:,} generations. "
                f"Final MSE: {fitness_history[-1]:.2f}"
            )
