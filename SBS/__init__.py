"""
SBS — Showcase by Science package.

Exposes the four modules that together implement the Evolutionary Image
Reconstruction system:

    dna_structures      — Genome and Triangle data structures (NumPy matrix)
    evolutionary_engine — GA loop: selection, crossover, mutation, elitism
    image_utils         — Rendering, edge detection, fitness functions
    dashboard_controller— Streamlit UI and session-state orchestration

Import convention used throughout the project:
    from SBS.dashboard_controller import ShowcaseDashboard
"""
