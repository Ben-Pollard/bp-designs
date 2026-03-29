"""Experiment demonstrating composition of multiple trees with borders."""

import random
from functools import partial
from pathlib import Path

import numpy as np

from bp_designs.core.color import Color
from bp_designs.core.geometry import Canvas
from bp_designs.core.lighting import DirectionalLighting
from bp_designs.core.scene import Scene
from bp_designs.generators.branching import strategies
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.primitives.two_d import Oval, Rectangle
from bp_designs.patterns.network import distribution
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

    # global_width = 2100
    # global_height = 2970
    global_width = 2032
    global_height = 2540
    global_canvas = Canvas.from_width_height(width=global_width, height=global_height)
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
            'cell': (0, 0), #top left
            # 'final_boundary': partial(lambda local_canvas: Oval.from_width_height(width=0.85, height=0.85, canvas=local_canvas).generate_pattern()),
            'final_boundary': partial(lambda local_canvas: Oval.from_bbox([0.1, 0.075, 0.9, 0.75], canvas=local_canvas).generate_pattern()),
            'organ': leaf_gen.generate_pattern(base_color=Color.from_hex("#075F3F"), scale=70.0),
            'growth_strategy': strategies.MomentumGrowth(segment_length=4, momentum=0.68),
            'attraction_strategy': None,
            'topology_strategy': None,
            'refinement_strategy': NetworkRefinementStrategy(
                decimate_min_distance=5,
                subdivide=False,
                relocate_alpha=0.2,
                relocate_iterations=0
            ),
        },
                {
            'cell': (0, 1), #top right
            # 'final_boundary': partial(lambda local_canvas: Oval.from_width_height(width=0.85, height=0.85, canvas=local_canvas).generate_pattern()),
            'final_boundary': partial(lambda local_canvas: Oval.from_bbox([0.1, 0.075, 0.9, 0.75], canvas=local_canvas).generate_pattern()),
            'growth_strategy': strategies.GridSnappedGrowth(segment_length=20, angles=4),
            'attraction_strategy': None,
            'topology_strategy': None,
            'refinement_strategy': NetworkRefinementStrategy(
                decimate_min_distance=3,
                subdivide=False,
                relocate_alpha=0.2,
                relocate_iterations=1
            ),
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
            'organ_distribution': distribution.RhythmicDistribution(interval=5, random_chance=0.8, min_depth_ratio=0.25)
        },

        {
            'cell': (1, 0), #bottom left
            # 'final_boundary': partial(lambda local_canvas: Oval.from_width_height(width=0.85, height=0.85, canvas=local_canvas).generate_pattern()),
            'final_boundary': partial(lambda local_canvas: Oval.from_bbox([0.1, 0.075, 0.9, 0.75], canvas=local_canvas).generate_pattern()),
            'growth_strategy': strategies.ObstacleAvoidanceGrowth(obstacles=[Oval.from_width_height(0.25, 0.25)]),
            # 'attraction_strategy': strategies.DriftAttraction(drift_vector=np.array([1,1])),
            'topology_strategy': None,
            'refinement_strategy': NetworkRefinementStrategy(
                decimate_min_distance=8,
                subdivide=True,
                relocate_alpha=0.25,
                relocate_iterations=10
            ),
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
            'cell': (1, 1), #bottom right
            # 'final_boundary': partial(lambda local_canvas: Oval.from_width_height(width=0.7, height=0.7, canvas=local_canvas).generate_pattern()),
            'final_boundary': partial(lambda local_canvas: Oval.from_bbox([0.1, 0.075, 0.9, 0.75], canvas=local_canvas).generate_pattern()),
            'organ': leaf_gen.generate_pattern(base_color=Color.from_hex("#F49F0A"), scale=70.0),
            'growth_strategy': None,
            # 'attraction_strategy': strategies.VortexAttraction(center=np.array([0.5,0.5]), strength=8),
            'topology_strategy': None,
            'refinement_strategy': NetworkRefinementStrategy(
                decimate_min_distance=5,
                subdivide=False,
                relocate_alpha=0.2,
                relocate_iterations=1
            ),
        },

    ]

    for p in unique_params:
        i, j = p['cell']
        organ = p['organ']

        # Local scene using sub_canvas for placement
        side_border = 90
        internal_gap = 45
        num_cols, num_rows = 2, 2

        cell_x_size = (global_width - (2 * side_border) - (num_cols - 1) * internal_gap) / num_cols
        cell_y_size = 1157

        v_offset = (global_canvas.height - (num_rows * cell_y_size + (num_rows - 1) * internal_gap)) / 2

        local_canvas = global_canvas.sub_canvas(
            x=side_border + j * (cell_x_size + internal_gap),
            y=v_offset + i * (cell_y_size + internal_gap),
            width=cell_x_size,
            height=cell_y_size
        )



        # Create generator
        gen = SpaceColonization(
            canvas=local_canvas,
            # root_position=PointPattern(0.5, 0.5, is_relative=True),
            root_position=PointPattern(0.5, 0.97, is_relative=True),
            initial_boundary=p['final_boundary'](local_canvas),
            final_boundary=p['final_boundary'](local_canvas),
            num_attractions=50,
            segment_length=5,
            kill_distance=15,
            max_iterations=500,
            seed=random.randint(0, 100),
            refinement_strategy=p.get('refinement_strategy'),
            growth_strategy=p['growth_strategy'],
            attraction_strategy=p.get('attraction_strategy'),
            topology_strategy=p['topology_strategy'],
            organ_template=p['organ'],
            organ_distribution=distribution.ClusterDistribution(count=5, random_chance=0.4, min_depth_ratio=0.25)
            # organ_distribution=distribution.RhythmicDistribution(interval=15, random_chance=0.8, min_depth_ratio=0.25) if not p.get('organ_distribution') else p['organ_distribution']
        )

        # Generate tree pattern
        network_style = NetworkStyle(
            distribution_params={"min_depth_ratio": 0.33, "random_chance": 0.25},
            render_mode="clipped_skin",
            taper_style="smooth",
            shading="linear",
            color_strategy="depth",
            thickness_mode="descendant", #hierarchy
            taper_power=0.5,
            max_thickness=cell_x_size / 20,
            min_thickness=1,
            start_color="#1F1E1E",
            end_color="#717171"
            # start_color=Color.from_hex(bg_dict[organ.base_color.to_hex().upper()]).with_hsl(s=1, lightness=0.05),
            # end_color=Color.from_hex(bg_dict[organ.base_color.to_hex().upper()]).with_hsl(s=0.5, lightness=0.5),
        )
        tree = gen.generate_pattern(**network_style.model_dump())


        tree_scene = Scene(local_canvas)

        rect_gen = Rectangle.from_canvas(local_canvas)
        tree_scene.add_layer(
            f"bg_{i}_{j}",
            rect_gen.generate_pattern(),
            fill=Color.from_hex(bg_dict[organ.base_color.to_hex().upper()]).with_hsl(s=0.15, lightness=0.85)
        )

        # Add tree to local scene
        tree_scene.add_layer(f"tree_{i}_{j}", tree)

        # 3. Add border for this slot
        tree_scene.add_layer(
            f"border_{i}_{j}",
            rect_gen.generate_pattern(),
            stroke_color="#1B1B1B",
            stroke_width=cell_x_size / 50,
            fill="none"
        )

        # 4. Add the local scene to the global scene
        scene.add_layer(
            f"slot_{i}_{j}",
            tree_scene
        )

    output_dir = Path("output/composition")
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "circular_grid" / "10x8_3.svg"
    svg_string = scene.to_svg()
    svg_path.write_text(svg_string)
    print(f"Saved composition to {svg_path}")
