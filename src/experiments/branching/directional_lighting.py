"""Experiment demonstrating the new directional lighting system."""

import numpy as np

from bp_designs.core.geometry import Canvas
from bp_designs.core.lighting import DirectionalLighting
from bp_designs.experiment.params import ParameterSpace, split_params
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.primitives.two_d import Oval
from bp_designs.patterns.organs import LeafGenerator
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
    """Run directional lighting exploration."""
    ref_canvas = Canvas.from_size(100)

    # Simple oval boundary
    oval = Oval.from_width_height(80.0, 80.0, canvas=ref_canvas, name="Circular boundary")
    boundary = oval.generate_pattern()

    root_pos = PointPattern(0.5, 1.0, is_relative=True, name="Bottom Center")

    # Create organ template
    leaf_gen = LeafGenerator()
    leaf_template = leaf_gen.generate_pattern(base_color="#4a7c44", scale=8.0)

    space = ParameterSpace(
        name="directional_lighting",
        specs={
            # Network structural parameters
            "network.num_attractions": [300],
            "network.segment_length": [2.0],
            "network.kill_distance": [5.0],
            "network.organ_template": [leaf_template],
            "network.organ_distribution": ["terminal"],
            "network.seed": 42,
            "network.canvas": [ref_canvas],
            "network.initial_boundary": [boundary],
            "network.final_boundary": [boundary],
            "network.root_position": [root_pos],
            # Rendering parameters
            "render.render_mode": "polygon",
            "render.thickness": "descendant",
            "render.min_thickness": 0.5,
            "render.max_thickness": 4.0,
            "render.color_strategy": "depth",
            "render.start_color": "#4a2c2a",
            "render.end_color": "#2d5a27",
            "render.background_color": "#f0f0e0",
            # Lighting parameters
            "render.light_angle": [45, 135, 225, 315],
            "render.highlight_amount": [0.1, 0.2],
            "render.shadow_amount": [0.1, 0.2],
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
        experiment_name="directional_lighting_exploration",
        svg_width=100,
        svg_height=100,
    )

    runner.run(grid=grid, generator_fn=generate_pattern)


if __name__ == "__main__":
    main()
