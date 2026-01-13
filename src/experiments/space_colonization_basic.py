"""Basic space colonization parameter exploration.

Explores key parameters: attraction_distance, num_attractions, segment_length.
"""


from bp_designs.core.geometry import Canvas
from bp_designs.experiment.params import ParameterGrid, ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.primitives.two_d import Oval
from bp_designs.patterns.shape import PointPattern


def generate_pattern(params: dict):
    """Generate pattern from parameters."""
    gen = SpaceColonization(**params)
    # Generate pattern using stored parameters
    network = gen.generate_pattern()
    return network


def main():
    """Run basic space colonization exploration."""
    # Create a reference canvas for defining relative patterns
    ref_canvas = Canvas.from_size(100)

    # Create multiple oval boundaries using different construction methods
    # These will be relative patterns because we pass ref_canvas
    boundaries = []

    # 1. Width/height construction (centered in canvas)
    oval1 = Oval.from_width_height(80.0, 60.0, canvas=ref_canvas, name="Medium centered oval")
    boundaries.append(oval1.generate_pattern())

    # 2. Tall narrow oval using bbox
    oval2 = Oval.from_bbox([20.0, 10.0, 80.0, 90.0], canvas=ref_canvas, name="Tall narrow oval")
    boundaries.append(oval2.generate_pattern())

    # 3. Wide flat oval using bbox
    oval3 = Oval.from_bbox([10.0, 40.0, 90.0, 60.0], canvas=ref_canvas, name="Wide flat oval")
    boundaries.append(oval3.generate_pattern())

    # 4. Small centered oval using bbox
    oval4 = Oval.from_bbox([35.0, 35.0, 65.0, 65.0], canvas=ref_canvas, name="Small centered oval")
    boundaries.append(oval4.generate_pattern())

    # 5. Large oval nearly filling canvas using width/height
    oval5 = Oval.from_width_height(95.0, 95.0, canvas=ref_canvas, name="Large filling oval")
    boundaries.append(oval5.generate_pattern())

    # Define possible root positions as relative PointPatterns
    root_positions = [
        PointPattern(0.5, 1.0, is_relative=True, name="Bottom Center"),
        PointPattern(0.25, 1.0, is_relative=True, name="Bottom Left"),
        PointPattern(0.75, 1.0, is_relative=True, name="Bottom Right"),
    ]

    # Define canvases of different sizes to explore density
    canvases = [
        Canvas.from_size(50),
        Canvas.from_size(100),
        Canvas.from_size(200),
    ]

    # Define parameter space with pattern and render parameters
    space = ParameterSpace(
        name="space_colonization_density_search",
        pattern={
            "num_attractions": [500],
            "segment_length": [2.0],
            "initial_boundary": boundaries,
            "final_boundary": boundaries,
            "root_position": root_positions,
            "kill_distance": [5.0],
            "seed": 42,
            "canvas": canvases,  # Varying canvas size explores density
            "max_iterations": 1000,
        },
        render={
            "thickness": "descendant",
            "min_thickness": [0.1, 0.5],  # 2 values
            "max_thickness": 5.0,
            "taper_power": [0.2, 0.5, 0.8],
            "thickness_mode": ["all_nodes", "leaves_only"],
            "render_mode": ["polyline", "polygon"],
            "taper_style": "smooth",
            "color": "black",
            "stroke_linecap": "round",
            "stroke_linejoin": "round",
            "width": 100,
            "height": 100,
            "padding": [0, 10],
        },
    )

    # Generate full grid
    full_grid = space.to_grid()
    print(f"Full grid size: {len(full_grid)} combinations")

    # Sample from grid to keep experiment tractable
    # Use systematic sampling: take every nth combination
    # Use random sampling to ensure all parameter combinations (like canvas size) are represented
    import random
    n_samples = 30
    random.seed(42)
    sampled_indices = random.sample(range(len(full_grid)), min(n_samples, len(full_grid)))
    sampled_indices.sort()
    sampled_grid = ParameterGrid(
        space_name=f"{space.name}_sampled",
        pattern_param_names=full_grid.pattern_param_names,
        render_param_names=full_grid.render_param_names,
        combinations=[full_grid[i] for i in sampled_indices],
    )

    print(f"Sampled grid size: {len(sampled_grid)} combinations")
    print(sampled_grid.summary())

    # Run experiment
    runner = ExperimentRunner(
        experiment_name="space_col_oval_variants",
        svg_width=100,
        svg_height=100,
        stroke_width=None,
    )

    runner.run(grid=sampled_grid, generator_fn=generate_pattern)


if __name__ == "__main__":
    main()
