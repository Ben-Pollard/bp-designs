"""Experiment demonstrating coupled parameters and new color system."""


from bp_designs.core.color import Color, Palette
from bp_designs.core.geometry import Canvas
from bp_designs.experiment.params import ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.primitives.two_d import Oval
from bp_designs.patterns.organs import LeafGenerator
from bp_designs.patterns.shape import PointPattern


def generate_pattern(params: dict):
    """Generate pattern from parameters."""
    # Use the new helper to automatically split structural and rendering parameters.
    # This will create the SpaceColonization generator with structural params
    # and store the remaining params in the resulting Pattern's render_params.
    return SpaceColonization.create_and_generate(params)


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

    def get_complementary_bg(organ_template):
        """Helper to make complimentary color logic explicit."""
        return Palette(organ_template.base_color).get_background().to_hex()

    space = ParameterSpace(
        name="coupled_organic_colors",
        pattern={
            "num_attractions": [300],
            "segment_length": [2.0],
            "kill_distance": [5.0],
            "organ_template": leaf_templates,
            "organ_distribution": ["terminal"],
            "seed": 42,
            "canvas": [ref_canvas],
            "initial_boundary": [boundary],
            "final_boundary": [boundary],
            "root_position": [root_pos],
        },
        render={
            "render_mode": "polygon",
            "thickness": "descendant",
            "min_thickness": 0.2,
            "max_thickness": 3.0,
            "color_strategy": "depth",
            "shading": ["linear", None],
            # These will be populated by derived parameters
            "background_color": [None],
            "start_color": [None],
            "end_color": [None],
        },
        derived={
            # Couple background color to organ color (complimentary)
            "background_color": lambda p: get_complementary_bg(p["organ_template"]),
            # Couple start/end colors for depth strategy to organ color
            "start_color": lambda p: p["organ_template"].base_color.darken(0.2).to_hex(),
            "end_color": lambda p: p["organ_template"].base_color.lighten(0.2).to_hex(),
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
