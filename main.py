"""
main.py — Application entry point.

This file is intentionally thin. Its sole responsibility is to instantiate
the top-level ShowcaseDashboard controller and hand control to Streamlit.
All UI layout, session state management, and EA orchestration live in
SBS/dashboard_controller.py; genetic data structures are in
SBS/dna_structures.py; the evolutionary algorithm core is in
SBS/evolutionary_engine.py; and rendering + fitness utilities are in
SBS/image_utils.py.

Architecture overview
---------------------
main.py
 └── ShowcaseDashboard          (SBS/dashboard_controller.py)
      ├── ImageReconstructor     (SBS/evolutionary_engine.py)
      │    └── Genome            (SBS/dna_structures.py)
      └── render_genome_fast /
          compute_fitness_weighted (SBS/image_utils.py)
"""

from SBS.dashboard_controller import ShowcaseDashboard

"""
Scientific Showcase Project: Evolutionary Image Reconstruction
Developed for the JOSEPHS Exhibition.
Disclaimer: The architecture, genetic algorithm logic, and Streamlit UI of
this project were co-developed with Artificial Intelligence (Claude 3.5
Sonnet / Gemini) to demonstrate the capabilities of AI-assisted software
engineering.
"""


def main() -> None:
    """
    Instantiate the Streamlit dashboard and execute one full rerun cycle.

    Streamlit re-calls this function from the top on every user interaction
    (button click, slider drag, etc.).  ShowcaseDashboard.__init__ runs
    st.set_page_config() — which must be the very first Streamlit call —
    before run() drives the sidebar, layout, and (if active) the EA loop.
    """
    app = ShowcaseDashboard()
    app.run()


if __name__ == "__main__":
    main()
