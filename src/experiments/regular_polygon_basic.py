"""Basic regular polygon parameter exploration.

Explores key parameters: sides, radius, rotation.
"""

import numpy as np

from bp_designs.core.geometry import Canvas
from bp_designs.experiment.params import ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.primitives.two_d import RegularPolygon


def generate_pattern(params: dict):
    """Generate pattern from parameters."""
    # Extract parameters for RegularPolygon constructor
    gen = RegularPolygon(**params)
    # Generate pattern using stored parameters
    shape_pattern = gen.generate_pattern()
    return shape_pattern


def main():
    """Run basic regular polygon exploration."""
    # Create canvas for this experiment
    width = 100
    canvas = Canvas.from_size(width)

    # Define parameter space
    space = ParameterSpace(
        name="regular_polygon_basic",
        specs={
            "sides": [3, 4, 5, 6, 8],  # triangle, square, pentagon, hexagon, octagon
            "radius": [20.0, 30.0, 40.0],  # different sizes
            "rotation": [0.0, np.pi / 12, np.pi / 6],  # 0°, 15°, 30°
            "canvas": canvas,
            "center": (50.0, 50.0),  # center in canvas
            "name": None,  # auto-generated name
        },
    )

    # Generate grid
    grid = space.to_grid()
    print(f"Total combinations: {len(grid)}")  # 5 × 3 × 3 = 45

    # Run experiment
    runner = ExperimentRunner(
        experiment_name="regular_polygon_basic_001",
        svg_width=100,
        svg_height=100,
        stroke_width=0.3,
    )

    runner.run(grid=grid, generator_fn=generate_pattern)


if __name__ == "__main__":
    main()
