"""Experiment demonstrating blossoms on branching networks."""

import numpy as np

from bp_designs.core.color import Color
from bp_designs.core.geometry import Canvas
from bp_designs.core.lighting import DirectionalLighting
from bp_designs.experiment.params import ParameterSpace, split_params
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.primitives.two_d import Oval
from bp_designs.patterns.network.refinement import NetworkRefinementStrategy
from bp_designs.patterns.organs import BlossomGenerator, LeafGenerator
from bp_designs.patterns.shape import PointPattern


def generate_pattern(params: dict):
    """Generate pattern from parameters."""
    network_params = split_params(params, "network")
    render_params = split_params(params, "render")

    # Create generator with structural params
    gen = SpaceColonization(**network_params)


    # Generate pattern with refinement and rendering params
    return gen.generate_pattern(**render_params)


def main():
    """Run blossom tree exploration."""
    ref_canvas = Canvas.from_size(1000)#.with_background_color("#d2dbaf")

    boundaries = []

    # 1. Width/height construction (centered in canvas)
    oval1 = Oval.from_bbox([0.1, 0.1, 0.9, 0.8], canvas=ref_canvas, name="Wide oval above center")
    boundaries.append(oval1.generate_pattern())


    # Define possible root positions as relative PointPatterns
    root_positions = [
        PointPattern(0.5, 1.0, is_relative=True, name="Bottom Center"),
        # PointPattern(0.45, 1.0, is_relative=True, name="Bottom Left"),
        # PointPattern(0.55, 1.0, is_relative=True, name="Bottom Right"),
    ]


    # Simple oval boundary
    oval = Oval.from_width_height(0.7, 0.75, canvas=ref_canvas, name="Circular boundary")
    boundary = oval.generate_pattern()

    root_pos = PointPattern(0.5, 1.0, is_relative=True, name="Bottom Center")

    leaf_gen = LeafGenerator()
    leaf_template = leaf_gen.generate_pattern(base_color=Color.from_hex("#4a7c44"), scale=80.0)

    blossom_gen = BlossomGenerator()
    blossom_templates = [blossom_gen.generate_pattern(
        base_color=Color.from_hex(b),
        scale=4.0,
        num_rings=2,
        inner_radius=0.0,
        ring_spacing=1.0,
        petal_shape="teardrop",
        center_color=c,
        jitter=0.1,
        overlap=1.2
    ) for b,c in [("#ffb7c5","#ffcc00"), ("#ffcc00", "#ffb7c5")]]


    space = ParameterSpace(
        name="big_trees",
        specs={
            # Network structural parameters
            "network.num_attractions": [10],
            "network.segment_length": [2.5],
            "network.kill_distance": [2],
            "network.seed": [42],
            "network.canvas": [ref_canvas],
            "network.max_iterations": [500],
            "network.initial_boundary": boundaries,
            "network.final_boundary": boundaries,
            "network.root_position": root_positions,
            "refinement.dist": [15],
            "refinement.subdivide": [True],
            "refinement.alpha": [0.5],
            "refinement.it": [3],
            "network.organ_template": blossom_templates + [leaf_template],
            "network.organ_distribution": ["terminal"],
            "render.distribution_params": [
                {"min_depth_ratio": 0.33, "random_chance": 0.5},
            ],
            # Rendering parameters
            "render.render_mode": "polyline",
            "render.taper_style": "smooth",
            "render.shading": "linear",
            "render.color_strategy": "depth",
            "render.thickness": "descendant",
            "render.start_color": "#3d2b1f",
            "render.end_color": "#5d4037",
            # Lighting parameters
            "render.light_angle": [45],
            "render.highlight_amount": [0.2],
            "render.shadow_amount": [0.2],
            "render.bg_saturation": [0.5],
            "render.bg_lightness": [0.94]
        },
        derived={
            # Create the LightingModel object from parameters
            "render.background_color": lambda p: p["network.organ_template"]
            .base_color.complementary()
            .with_hsl(s=p["render.bg_saturation"], lightness=p["render.bg_lightness"]),
            "render.lighting": lambda p: DirectionalLighting(
                light_direction=np.array([
                    np.cos(np.radians(p["render.light_angle"])),
                    np.sin(np.radians(p["render.light_angle"]))
                ]),
                highlight_amount=p["render.highlight_amount"],
                shadow_amount=p["render.shadow_amount"]
            ),
            "network.refinement_strategy": lambda p: NetworkRefinementStrategy(
        decimate_min_distance=p['refinement.dist'], subdivide=p['refinement.subdivide'], relocate_alpha=p['refinement.alpha'], relocate_iterations=p['refinement.it']),
        "render.max_thickness": lambda p: p['network.canvas'].width / 20,
        "render.min_thickness": lambda p: 1.0

        },
    )

    grid = space.to_grid()

    runner = ExperimentRunner(
        experiment_name="big_trees",
        svg_width=1000,
        svg_height=1000,
    )

    runner.run(grid=grid, generator_fn=generate_pattern, max_variants=50, parallel=True)


if __name__ == "__main__":
    main()
