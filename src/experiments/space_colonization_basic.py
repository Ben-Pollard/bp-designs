"""Basic space colonization parameter exploration.

Explores key parameters: attraction_distance, num_attractions, segment_length.
"""

from bp_designs.experiment.params import ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.branching.space_colonization import SpaceColonization


def generate_pattern(params: dict):
    """Generate pattern from parameters."""
    gen = SpaceColonization(**params)
    return gen.generate_pattern().to_geometry()


def main():
    """Run basic space colonization exploration."""
    # Define parameter space
    space = ParameterSpace(
        name="space_colonization_basic",
        ranges={
            "num_attractions": [20, 50, 100],  # 3 values
            "segment_length": [1.0, 2.0, 4.0],  # 3 values
        },
        fixed={
            "seed": 42,  # Same seed for fair comparison
            "kill_distance": 5.0,
            "width": 100.0,
            "height": 100.0,
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
