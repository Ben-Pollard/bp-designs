"""Space colonization seed variations.

Explores pattern diversity across different random seeds with fixed parameters.
"""

from bp_designs.experiment import ExperimentRunner, ParameterSpace
from bp_designs.generators.branching.space_colonization import SpaceColonization


def generate_pattern(params: dict):
    """Generate pattern from parameters."""
    gen = SpaceColonization(**params)
    return gen.generate_pattern().to_geometry()


def main():
    """Run seed variation exploration."""
    # Define parameter space
    space = ParameterSpace(
        name="space_colonization_seeds",
        ranges={
            "seed": list(range(20)),  # 20 different seeds
        },
        fixed={
            "num_attractions": 500,
            "attraction_distance": 50.0,
            "kill_distance": 5.0,
            "segment_length": 2.0,
            "width": 100.0,
            "height": 100.0,
        },
    )

    # Generate grid
    grid = space.to_grid()
    print(f"Total variations: {len(grid)}")  # 20

    # Run experiment
    runner = ExperimentRunner(
        experiment_name="space_col_seeds_001",
        svg_width=100,
        svg_height=100,
        stroke_width=0.3,
    )

    runner.run(grid=grid, generator_fn=generate_pattern)


if __name__ == "__main__":
    main()
