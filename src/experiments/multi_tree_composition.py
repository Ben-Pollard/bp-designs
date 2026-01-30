"""Experiment demonstrating composition of multiple trees with borders."""

from random import randint

import numpy as np

from bp_designs.core.color import Color
from bp_designs.core.geometry import Canvas
from bp_designs.core.lighting import DirectionalLighting
from bp_designs.core.scene import Scene
from bp_designs.experiment.params import ParameterSpace, split_params
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.primitives.two_d import Oval, Rectangle
from bp_designs.patterns.network.refinement import NetworkRefinementStrategy
from bp_designs.patterns.organs import BlossomGenerator, LeafGenerator
from bp_designs.patterns.shape import PointPattern


def generate_pattern(params: dict):
    """Generate a composition of multiple trees."""
    # This generator function will now return a Scene instead of a single BranchNetwork

    # Global canvas for the whole composition
    global_canvas = Canvas.from_size(2000)
    scene = Scene(global_canvas)

    # We'll create a 2x2 grid of trees
    grid_size = 2
    cell_size = 1000

    # Shared parameters for all trees (with some variation)
    network_params = split_params(params, "network")
    render_params = split_params(params, "render")
    refinement_params = split_params(params, "refinement")

    # Extract lighting for the whole scene
    lighting = render_params.get("lighting")

    for i in range(grid_size):
        for j in range(grid_size):
            # Local canvas for each tree "slot"
            local_canvas = Canvas.from_size(cell_size)

            # Create a local scene for this tree + its frame
            tree_scene = Scene(local_canvas)

            # 1. Add background for this slot
            rect_gen = Rectangle.from_canvas(local_canvas)
            bg_color = render_params.get("background_color", "#ffffff")
            tree_scene.add_layer(
                f"bg_{i}_{j}",
                rect_gen.generate_pattern(),
                fill=bg_color
            )

            # 2. Generate the tree
            # We need a unique seed for each tree in the grid
            tree_seed = network_params.get("seed", 0) + i * grid_size + j

            # Create a local copy of network params with the unique seed
            local_network_params = network_params.copy()
            local_network_params["seed"] = tree_seed
            local_network_params["canvas"] = local_canvas

            # Define boundaries for this local canvas
            oval = Oval.from_bbox([0.1, 0.1, 0.9, 0.8], canvas=local_canvas)
            boundary = oval.generate_pattern()
            local_network_params["initial_boundary"] = boundary
            local_network_params["final_boundary"] = boundary
            local_network_params["root_position"] = PointPattern(0.5, 1.0, is_relative=True)

            # Create generator
            gen = SpaceColonization(**local_network_params)

            # Generate tree pattern
            tree = gen.generate_pattern(**render_params)

            # Add tree to local scene
            tree_scene.add_layer(f"tree_{i}_{j}", tree)

            # 3. Add border for this slot
            tree_scene.add_layer(
                f"border_{i}_{j}",
                rect_gen.generate_pattern(),
                stroke_color="#333333",
                stroke_width=10,
                fill="none"
            )

            # 4. Add the local scene to the global scene with a transform
            tx = j * cell_size
            ty = i * cell_size
            scene.add_layer(
                f"slot_{i}_{j}",
                tree_scene,
                transform=f"translate({tx}, {ty})"
            )

    # Set the global lighting
    scene.render_params["lighting"] = lighting

    return scene


def main():
    """Run multi-tree composition exploration."""
    ref_canvas = Canvas.from_size(1000)

    leaf_gen = LeafGenerator()
    leaf_templates = [leaf_gen.generate_pattern(base_color=Color.from_hex(b), scale=70.0) for b in ["#2D936C", "#F08700"]]

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
    ) for b,c in [("#F4BBD3","#FEFA86"), ("#FEFA86", "#F4BBD3")]]

    bg_dict = {
        "#F4BBD3": "#FEFA86",
        "#FEFA86": "#F4BBD3",
        "#2D936C": "#F08700",
        "#F08700": "#F4BBD3"
    }

    space = ParameterSpace(
        name="multi_tree_composition",
        specs={
            "network.segment_length": [5.0],
            "network.kill_distance": [4.0],
            "network.max_iterations": [300],
            "refinement.dist": [15],
            "refinement.subdivide": [True],
            "refinement.alpha": [0.5],
            "refinement.it": [3],
            "network.organ_template": blossom_templates,
            "network.organ_distribution": ["terminal"],
            "render.distribution_params": [
                {"min_depth_ratio": 0.33, "random_chance": 0.5},
            ],
            "render.render_mode": "polygon",
            "render.taper_style": "smooth",
            "render.shading": "linear",
            "render.color_strategy": "depth",
            "render.thickness": "descendant",
            "render.start_color": "#694A38",
            "render.end_color": "#5B4B49",
            "render.light_angle": [45],
            "render.highlight_amount": [0.2],
            "render.shadow_amount": [0.2],
            "render.bg_saturation": [0.3],
            "render.bg_lightness": [0.95]
        },
        derived={
            "render.background_color": lambda p: Color.from_hex(bg_dict.get(p["network.organ_template"]
            .base_color.to_hex().upper()))
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
                decimate_min_distance=p['refinement.dist'],
                subdivide=p['refinement.subdivide'],
                relocate_alpha=p['refinement.alpha'],
                relocate_iterations=p['refinement.it']
            ),
            "render.max_thickness": lambda p: 50.0,
            "render.min_thickness": lambda p: 1.0,
            "network.seed": lambda p: randint(1, 1000),
            "network.num_attractions": lambda p: 15,
        },
    )

    grid = space.to_grid()

    runner = ExperimentRunner(
        experiment_name="multi_tree_composition",
        svg_width=1000,
        svg_height=1000,
    )

    runner.run(grid=grid, generator_fn=generate_pattern, max_variants=5, parallel=True)


if __name__ == "__main__":
    main()
