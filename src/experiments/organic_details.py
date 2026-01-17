"""Experiment to demonstrate organic details: organs (leaves/fruits) and color strategies."""

from bp_designs.core.geometry import Canvas
from bp_designs.experiment.params import ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.primitives.two_d import Oval
from bp_designs.patterns.network import BlossomOrgan, CircleOrgan, DetailedLeafOrgan, LeafOrgan, StarOrgan
from bp_designs.patterns.shape import PointPattern


def generate_pattern(params: dict):
    """Generate pattern from parameters and attach organs."""
    # Extract organ params from pattern params if they exist
    organ_type_name = params.pop("organ_type", "leaf")
    distribution_name = params.pop("distribution", "terminal")
    organ_scale = params.pop("organ_scale", 5.0)
    organ_color = params.pop("organ_color", "#4caf50")
    dist_count = params.pop("dist_count", 3)
    dist_interval = params.pop("dist_interval", 4)

    gen = SpaceColonization(**params)
    network = gen.generate_pattern()

    # Map organ type name to class
    organ_map = {
        "leaf": LeafOrgan,
        "blossom": BlossomOrgan,
        "star": StarOrgan,
        "detailed_leaf": DetailedLeafOrgan,
        "circle": CircleOrgan,
    }
    organ_type = organ_map.get(organ_type_name, LeafOrgan)

    # Prepare distribution params
    dist_params = {}
    if distribution_name == "cluster":
        dist_params["count"] = dist_count
    elif distribution_name == "rhythmic":
        dist_params["interval"] = dist_interval

    organ_kwargs = {"scale": organ_scale, "fill": organ_color}
    if organ_type_name == "leaf":
        organ_kwargs["jitter"] = 30.0

    network.attach_organs(
        organ_type, distribution=distribution_name, distribution_params=dist_params, **organ_kwargs
    )

    return network


def main():
    """Run organic details exploration."""
    ref_canvas = Canvas.from_size(100)

    # Simple oval boundary
    oval = Oval.from_width_height(80.0, 80.0, canvas=ref_canvas, name="Circular boundary")
    boundary = oval.generate_pattern()

    root_pos = PointPattern(0.5, 1.0, is_relative=True, name="Bottom Center")

    space = ParameterSpace(
        name="organic_details_exploration",
        specs={
            "num_attractions": [300],
            "segment_length": [3.0],
            "initial_boundary": [boundary],
            "final_boundary": [boundary],
            "root_position": [root_pos],
            "kill_distance": [6.0],
            "seed": [42],
            "canvas": [ref_canvas],
            "max_iterations": 500,
            "organ_type": ["blossom", "star", "detailed_leaf", "leaf"],
            "distribution": ["terminal", "cluster", "rhythmic"],
            "organ_scale": [5.0, 8.0],
            "organ_color": ["#ffb7c5", "#ffd700", "#2e7d32", "#4caf50"],
            "thickness": "descendant",
            "min_thickness": 0.5,
            "max_thickness": 4.0,
            "color_strategy": ["depth"],
            "render_mode": "polyline",
            "taper_style": "smooth",
            "width": 400,
            "height": 400,
            "padding": 20,
        },
    )

    grid = space.to_grid()
    print(f"Running experiment with {len(grid)} variants...")

    runner = ExperimentRunner(
        experiment_name="organic_details_demo",
        svg_width=400,
        svg_height=400,
    )

    runner.run(grid=grid, generator_fn=generate_pattern)


if __name__ == "__main__":
    main()
