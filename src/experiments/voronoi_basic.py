"""Basic Voronoi tessellation parameter exploration.

Explores key parameters: num_sites, relaxation_iterations, render_mode.
"""

from bp_designs.core.geometry import Canvas
from bp_designs.experiment.params import ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.cellular.voronoi import VoronoiTessellation


def generate_pattern(params: dict):
    """Generate pattern from parameters."""
    gen = VoronoiTessellation(**params)
    return gen.generate_pattern()


def main():
    """Run basic Voronoi tessellation exploration."""
    # Define parameter space
    space = ParameterSpace(
        name="voronoi_basic",
        specs={
            "num_sites": [10, 30, 50, 100],  # 4 values
            "relaxation_iterations": [0, 2, 5],  # 3 values
            "render_mode": ["edges", "cells", "both"],  # 3 values
            "seed": 42,  # Same seed for fair comparison
            "width": 100.0,
            "height": 100.0,
            "boundary_margin": 20.0,
            "canvas": Canvas.from_width_height(100, 100),
        },
    )

    # Generate grid
    grid = space.to_grid()
    print(f"Total combinations: {len(grid)}")  # 4 × 3 × 3 = 36

    # Run experiment
    runner = ExperimentRunner(
        experiment_name="voronoi_basic_001",
        svg_width=100,
        svg_height=100,
        stroke_width=0.3,
    )

    runner.run(grid=grid, generator_fn=generate_pattern)


if __name__ == "__main__":
    main()
