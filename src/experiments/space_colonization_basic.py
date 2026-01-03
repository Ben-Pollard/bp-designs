"""Basic space colonization parameter exploration.

Explores key parameters: attraction_distance, num_attractions, segment_length.
"""


from bp_designs.core.geometry import Canvas, Point
from bp_designs.experiment.params import ParameterGrid, ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.primitives.two_d import Oval


def generate_pattern(params: dict):
    """Generate pattern from parameters."""
    gen = SpaceColonization(**params)
    # Generate pattern using stored parameters
    network = gen.generate_pattern()
    return network.to_geometry()


def main():
    """Run basic space colonization exploration."""
    # Create geometry objects for this experiment
    width = 100
    height = 100
    canvas = Canvas.from_size(width)

    # Create multiple oval boundaries using different construction methods
    boundaries = []

    # 1. Width/height construction (centered in canvas)
    oval1 = Oval.from_width_height(80.0, 60.0, canvas=canvas, name="Medium centered oval")
    boundaries.append(oval1.generate_pattern())

    # 2. Tall narrow oval using bbox
    oval2 = Oval.from_bbox([20.0, 10.0, 80.0, 90.0], canvas=canvas, name="Tall narrow oval")
    boundaries.append(oval2.generate_pattern())

    # 3. Wide flat oval using bbox
    oval3 = Oval.from_bbox([10.0, 40.0, 90.0, 60.0], canvas=canvas, name="Wide flat oval")
    boundaries.append(oval3.generate_pattern())

    # 4. Small centered oval using bbox
    oval4 = Oval.from_bbox([35.0, 35.0, 65.0, 65.0], canvas=canvas, name="Small centered oval")
    boundaries.append(oval4.generate_pattern())

    # 5. Large oval nearly filling canvas using width/height
    oval5 = Oval.from_width_height(95.0, 95.0, canvas=canvas, name="Large filling oval")
    boundaries.append(oval5.generate_pattern())

    # Define possible root positions (bottom center, left center, right center)
    root_positions = [
        Point(x=int(width / 2), y=int(height), z=None),  # bottom center
        Point(x=int(width / 4), y=int(height), z=None),  # bottom left quarter
        Point(x=int(3 * width / 4), y=int(height), z=None),  # bottom right quarter
    ]

    # Define parameter space with all variable parameters
    space = ParameterSpace(
        name="space_colonization_oval_variants",
        ranges={
            "num_attractions": [1, 2, 3],  # 3 values
            "segment_length": [1.0, 2.0, 4.0],  # 3 values
            "initial_boundary": boundaries,  # 5 oval variants
            "final_boundary": boundaries,  # same as initial for now
            "root_position": root_positions,  # 3 positions
            "kill_distance": [3.0, 5.0, 8.0],  # 3 values
        },
        fixed={
            "seed": 42,  # Same seed for fair comparison
            "canvas": canvas,
            "max_iterations": 1000,
        },
    )

    # Generate full grid
    full_grid = space.to_grid()
    print(f"Full grid size: {len(full_grid)} combinations")

    # Sample from grid to keep experiment tractable
    # Use systematic sampling: take every nth combination
    sample_every = max(1, len(full_grid) // 50)  # Target ~50 samples
    sampled_grid = ParameterGrid(
        space_name=f"{space.name}_sampled",
        param_names=full_grid.param_names,
        fixed_params=full_grid.fixed_params,
        combinations=[full_grid[i] for i in range(0, len(full_grid), sample_every)],
    )

    print(f"Sampled grid size: {len(sampled_grid)} combinations")
    print(sampled_grid.summary())

    # Run experiment
    runner = ExperimentRunner(
        experiment_name="space_col_oval_variants",
        svg_width=100,
        svg_height=100,
        stroke_width=0.3,
    )

    runner.run(grid=sampled_grid, generator_fn=generate_pattern)


if __name__ == "__main__":
    main()
