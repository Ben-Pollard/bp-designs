"""Experiment demonstrating composition of multiple trees with borders."""

import random
from pathlib import Path

import numpy as np

from bp_designs.core.color import Color
from bp_designs.core.geometry import Canvas
from bp_designs.core.lighting import DirectionalLighting
from bp_designs.core.scene import Scene
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.primitives.two_d import Oval, Rectangle
from bp_designs.patterns.network.refinement import NetworkRefinementStrategy
from bp_designs.patterns.network.renderer import NetworkStyle
from bp_designs.patterns.organs import BlossomGenerator, LeafGenerator
from bp_designs.patterns.shape import PointPattern

if __name__ == "__main__":
    """Run multi-tree composition."""

    # Global canvas for the whole composition
    global_canvas = Canvas.from_size(2000)

    light_angle = 45
    lighting = DirectionalLighting(
        light_direction=np.array([
            np.cos(np.radians(light_angle)),
            np.sin(np.radians(light_angle))
        ]),
        highlight_amount=0.5,
        shadow_amount=0.5
    )
    scene = Scene(global_canvas, render_params={"lighting": lighting})

    # We'll create a 2x2 grid of trees


    bg_dict = {
        "#F4BBD3": "#FEFA86",
        "#FEFA86": "#F4BBD3",
        "#2D936C": "#F08700",
        "#F08700": "#F4BBD3"
    }

    leaf_gen = LeafGenerator()
    blossom_gen = BlossomGenerator()

    unique_params = [
        {
            'cell': (0, 0),
            'organ': leaf_gen.generate_pattern(base_color=Color.from_hex("#2D936C"), scale=70.0),
        },
        {
            'cell': (1, 0),
            'organ': leaf_gen.generate_pattern(base_color=Color.from_hex("#F08700"), scale=70.0),
        },
        {
            'cell': (1, 1),
            'organ': blossom_gen.generate_pattern(
                base_color=Color.from_hex("#F4BBD3"),
                scale=4.0,
                num_rings=2,
                inner_radius=0.0,
                ring_spacing=1.0,
                petal_shape="teardrop",
                center_color="#FEFA86",
                jitter=0.1,
                overlap=1.2
            ),
        },
        {
            'cell': (0, 1),
            'organ': blossom_gen.generate_pattern(
                base_color=Color.from_hex("#FEFA86"),
                scale=4.0,
                num_rings=2,
                inner_radius=0.0,
                ring_spacing=1.0,
                petal_shape="teardrop",
                center_color="#F4BBD3",
                jitter=0.1,
                overlap=1.2
            ),
        },
    ]

    for p in unique_params:
        i, j = p['cell']
        organ = p['organ']

        # Local scene using sub_canvas for placement
        cell_size = 1000
        local_canvas = global_canvas.sub_canvas(
            x=j * cell_size,
            y=i * cell_size,
            width=cell_size,
            height=cell_size
        )
        tree_scene = Scene(local_canvas)

        # Define boundaries for this local canvas
        oval = Oval.from_bbox([0.1, 0.1, 0.9, 0.8], canvas=local_canvas)
        boundary = oval.generate_pattern()

        # Create generator
        gen = SpaceColonization(
            canvas=local_canvas,
            root_position=PointPattern(0.5, 0.98, is_relative=True),
            initial_boundary=boundary,
            final_boundary=boundary,
            num_attractions=random.randint(10, 20),
            segment_length=5,
            kill_distance=4,
            max_iterations=300,
            seed=random.randint(0, 100),
            refinement_strategy=NetworkRefinementStrategy(
                decimate_min_distance=random.randint(5, 15),
                subdivide=True,
                relocate_alpha=0.5,
                relocate_iterations=3
            ),
            organ_template=organ,
            organ_distribution="terminal"
        )

        # Generate tree pattern
        network_style = NetworkStyle(
            distribution_params={"min_depth_ratio": 0.33, "random_chance": 0.5},
            render_mode="polygon",
            taper_style="smooth",
            shading="linear",
            color_strategy="depth",
            thickness="descendant",
            max_thickness=cell_size / 20,
            min_thickness=1,
            start_color="#694A38",
            end_color="#5B4B49"
        )
        tree = gen.generate_pattern(**network_style.model_dump())

        rect_gen = Rectangle.from_canvas(local_canvas)
        tree_scene.add_layer(
            f"bg_{i}_{j}",
            rect_gen.generate_pattern(),
            fill=Color.from_hex(bg_dict[organ.base_color.to_hex().upper()]).with_hsl(s=0.2, lightness=0.5)
        )

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

        # 4. Add the local scene to the global scene
        scene.add_layer(
            f"slot_{i}_{j}",
            tree_scene
        )

    output_dir = Path("output/composition")
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "4trees.svg"
    svg_string = scene.to_svg()
    svg_path.write_text(svg_string)
    print(f"Saved composition to {svg_path}")
