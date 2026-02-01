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


    light_angle = 45
    lighting = DirectionalLighting(
        light_direction=np.array([
            np.cos(np.radians(light_angle)),
            np.sin(np.radians(light_angle))
        ]),
        highlight_amount=0.2,
        shadow_amount=0.05
    )

    global_canvas = Canvas.from_width_height(width=2100, height=2970)
    scene = Scene(global_canvas, render_params={"lighting": lighting})

    scene.add_layer(
        "bg_global",
        Rectangle.from_canvas(global_canvas).generate_pattern(),
        fill=Color.from_hex("#ffffff")
    )


    # We'll create a 2x2 grid of trees


    bg_dict = {
        "#E359D1": "#075F3F",
        "#C061FB": "#F49F0A",
        "#075F3F": "#E359D1",
        "#F49F0A": "#C061FB"
    }

    leaf_gen = LeafGenerator()
    blossom_gen = BlossomGenerator()

    unique_params = [
        {
            'cell': (0, 0),
            'organ': leaf_gen.generate_pattern(base_color=Color.from_hex("#075F3F"), scale=70.0),
        },
        {
            'cell': (1, 1),
            'organ': leaf_gen.generate_pattern(base_color=Color.from_hex("#F49F0A"), scale=70.0),
        },
        {
            'cell': (1, 0),
            'organ': blossom_gen.generate_pattern(
                base_color=Color.from_hex("#E359D1"),
                scale=5.0,
                num_rings=2,
                inner_radius=0.0,
                ring_spacing=1.0,
                petal_shape="teardrop",
                center_color="#F49F0A",
                jitter=0.1,
                overlap=1.2
            ),
        },
        {
            'cell': (0, 1),
            'organ': blossom_gen.generate_pattern(
                base_color=Color.from_hex("#C061FB"),
                scale=5.0,
                num_rings=2,
                inner_radius=0.0,
                ring_spacing=1.0,
                petal_shape="teardrop",
                center_color="#F49F0A",
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
        num_cols, num_rows = 2, 2
        h_gap = (global_canvas.width - num_cols * cell_size) / (num_cols + 1)
        v_gap = h_gap  # Vertical spacing same as horizontal
        v_offset = (global_canvas.height - (num_rows * cell_size + (num_rows - 1) * v_gap)) / 2

        local_canvas = global_canvas.sub_canvas(
            x=h_gap + j * (cell_size + h_gap),
            y=v_offset + i * (cell_size + v_gap),
            width=cell_size,
            height=cell_size
        )


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
                decimate_min_distance=random.randint(5, 10),
                subdivide=True,
                relocate_alpha=0.2,
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


        tree_scene = Scene(local_canvas)

        rect_gen = Rectangle.from_canvas(local_canvas)
        tree_scene.add_layer(
            f"bg_{i}_{j}",
            rect_gen.generate_pattern(),
            fill=Color.from_hex(bg_dict[organ.base_color.to_hex().upper()]).with_hsl(s=0.2, lightness=0.9)
        )

        # Add tree to local scene
        tree_scene.add_layer(f"tree_{i}_{j}", tree)

        # 3. Add border for this slot
        tree_scene.add_layer(
            f"border_{i}_{j}",
            rect_gen.generate_pattern(),
            stroke_color="#1B1B1B",
            stroke_width=cell_size / 50,
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
