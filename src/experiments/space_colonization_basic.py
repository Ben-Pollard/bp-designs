"""Basic space colonization parameter exploration.

Explores key parameters: attraction_distance, num_attractions, segment_length.
"""

import numpy as np

from bp_designs.core.geometry import Canvas, Point, Polygon
from bp_designs.experiment.params import ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.branching.space_colonization import SpaceColonization


def generate_pattern(params: dict):
    """Generate pattern from parameters."""
    gen = SpaceColonization(**params)
    # Generate pattern using stored parameters
    network = gen.generate_pattern()
    return network.to_geometry()


def main():
    """Run basic space colonization exploration."""
    # Create geometry objects for this experiment
    width = 100.0
    height = 100.0
    canvas_coords = np.array(
        [
            [0.0, 0.0],
            [width, 0.0],
            [width, height],
            [0.0, height],
            [0.0, 0.0],  # Close the polygon
        ]
    )
    canvas = Canvas(coords=canvas_coords)
    root_position = Point(x=int(width / 2), y=int(height), z=None)
    boundary = Polygon(coords=canvas_coords)  # Same polygon for initial and final boundaries

    # Define parameter space
    space = ParameterSpace(
        name="space_colonization_basic",
        ranges={
            "num_attractions": [1, 2, 3],  # 3 values
            "segment_length": [1.0, 2.0, 4.0],  # 3 values
        },
        fixed={
            "seed": 42,  # Same seed for fair comparison
            "kill_distance": 5.0,
            "canvas": canvas,
            "root_position": root_position,
            "initial_boundary": boundary,
            "final_boundary": boundary,
            "max_iterations": 1000,
        },
    )

    # Generate grid
    grid = space.to_grid()
    print(f"Total combinations: {len(grid)}")  # 3 × 3 × 3 = 27

    # Run experiment
    runner = ExperimentRunner(
        experiment_name="space_col_basic_001",
        svg_width=100,
        svg_height=100,
        stroke_width=0.3,
    )

    runner.run(grid=grid, generator_fn=generate_pattern)


if __name__ == "__main__":
    main()
