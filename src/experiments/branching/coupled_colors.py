"""Experiment demonstrating coupled parameters and new color system."""

from bp_designs.core.color import Color
from bp_designs.core.geometry import Canvas
from bp_designs.experiment.params import ParameterSpace, split_params
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.primitives.two_d import Oval
from bp_designs.patterns.organs import LeafGenerator
from bp_designs.patterns.shape import PointPattern


def generate_pattern(params: dict):
    """Generate pattern from parameters."""
    # Route namespaced parameters
    # 'network' namespace for SpaceColonization structural params
    # 'render' namespace for rendering parameters
    network_params = split_params(params, "network")
    render_params = split_params(params, "render")

    # Create generator with structural params
    gen = SpaceColonization(**network_params)

    # Generate pattern with rendering params
    return gen.generate_pattern(**render_params)


def main():
    """Run coupled parameter exploration."""
    ref_canvas = Canvas.from_size(100)

    # Simple oval boundary
    oval = Oval.from_width_height(80.0, 80.0, canvas=ref_canvas, name="Circular boundary")
    boundary = oval.generate_pattern()

    root_pos = PointPattern(0.5, 1.0, is_relative=True, name="Bottom Center")

    # Define a few base colors for organs
    organ_colors = [
        Color.from_hex("#2d5a27"),  # Forest Green
        Color.from_hex("#a52a2a"),  # Brown/Red
        Color.from_hex("#4b0082"),  # Indigo
    ]

    # Create organ templates - increased scale for visibility
    leaf_gen = LeafGenerator()
    leaf_templates = [leaf_gen.generate_pattern(base_color=c, scale=8.0) for c in organ_colors]

    space = ParameterSpace(
        name="coupled_organic_colors",
        specs={
            # Network structural parameters
            "network.num_attractions": [300],
            "network.segment_length": [2.0],
            "network.kill_distance": [5.0],
            "network.organ_template": leaf_templates,
            "network.organ_distribution": ["terminal"],
            "network.seed": 42,
            "network.canvas": [ref_canvas],
            "network.initial_boundary": [boundary],
            "network.final_boundary": [boundary],
            "network.root_position": [root_pos],
            # Rendering parameters
            "render.render_mode": "polygon",
            "render.thickness": "descendant",
            "render.min_thickness": 0.2,
            "render.max_thickness": 3.0,
            "render.color_strategy": "depth",
            "render.shading": ["linear", None],
            # Parameters for background derivation
            "render.bg_saturation": [0.1, 0.2, 0.5],
            "render.bg_lightness": [0.94, 0.98],
        },
        derived={
            # Couple background color to organ color (complimentary)
            "render.background_color": lambda p: p["network.organ_template"]
            .base_color.complementary()
            .with_hsl(s=p["render.bg_saturation"], l=p["render.bg_lightness"]),
            # Couple start/end colors for depth strategy to organ color
            "render.start_color": lambda p: p["network.organ_template"].base_color.darken(0.2),
            "render.end_color": lambda p: p["network.organ_template"].base_color.lighten(0.2),
        },
    )

    grid = space.to_grid()

    runner = ExperimentRunner(
        experiment_name="coupled_colors_test",
        svg_width=100,
        svg_height=100,
    )

    runner.run(grid=grid, generator_fn=generate_pattern)


if __name__ == "__main__":
    main()
