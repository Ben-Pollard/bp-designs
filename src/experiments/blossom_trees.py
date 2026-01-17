"""Experiment demonstrating blossoms on branching networks."""

import numpy as np

from bp_designs.core.geometry import Canvas
from bp_designs.core.lighting import DirectionalLighting
from bp_designs.experiment.params import ParameterSpace, split_params
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.primitives.two_d import Oval
from bp_designs.patterns.organs import BlossomGenerator
from bp_designs.patterns.shape import PointPattern


def generate_pattern(params: dict):
    """Generate pattern from parameters."""
    network_params = split_params(params, "network")
    render_params = split_params(params, "render")

    # Create generator with structural params
    gen = SpaceColonization(**network_params)

    # Generate pattern with rendering params
    return gen.generate_pattern(**render_params)


def main():
    """Run blossom tree exploration."""
    ref_canvas = Canvas.from_size(100)

    # Simple oval boundary
    oval = Oval.from_width_height(80.0, 80.0, canvas=ref_canvas, name="Circular boundary")
    boundary = oval.generate_pattern()

    root_pos = PointPattern(0.5, 1.0, is_relative=True, name="Bottom Center")

    # Create blossom template with "best" params
    blossom_gen = BlossomGenerator()
    blossom_template = blossom_gen.generate_pattern(
        base_color="#ffb7c5",  # Cherry blossom pink
        scale=1.0,
        num_rings=2,
        inner_radius=0.0,
        ring_spacing=1.0,
        petal_shape="teardrop",
        center_color="#ffcc00",
        jitter=0.1,
        overlap=1.2
    )

    space = ParameterSpace(
        name="blossom_trees",
        specs={
            # Network structural parameters
            "network.num_attractions": [200],
            "network.segment_length": [2.5],
            "network.kill_distance": [6.0],
            "network.organ_template": [blossom_template],
            "network.organ_distribution": ["terminal"],
            "network.seed": [42, 123],
            "network.canvas": [ref_canvas],
            "network.initial_boundary": [boundary],
            "network.final_boundary": [boundary],
            "network.root_position": [root_pos],
            # Rendering parameters
            "render.render_mode": "polygon",
            "render.thickness": "descendant",
            "render.min_thickness": 0.5,
            "render.max_thickness": 3.0,
            "render.color_strategy": "depth",
            "render.start_color": "#3d2b1f",
            "render.end_color": "#5d4037",
            "render.background_color": "#fdf5e6",
            # Lighting parameters
            "render.light_angle": [45, 225],
            "render.highlight_amount": [0.2],
            "render.shadow_amount": [0.2],
        },
        derived={
            # Create the LightingModel object from parameters
            "render.lighting": lambda p: DirectionalLighting(
                light_direction=np.array([
                    np.cos(np.radians(p["render.light_angle"])),
                    np.sin(np.radians(p["render.light_angle"]))
                ]),
                highlight_amount=p["render.highlight_amount"],
                shadow_amount=p["render.shadow_amount"]
            )
        },
    )

    grid = space.to_grid()

    runner = ExperimentRunner(
        experiment_name="blossom_trees_exploration",
        svg_width=100,
        svg_height=100,
    )

    runner.run(grid=grid, generator_fn=generate_pattern)


if __name__ == "__main__":
    main()
