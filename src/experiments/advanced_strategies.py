"""Experiment demonstrating advanced Space Colonization strategies: Obstacle Avoidance and Moving Attractors."""


import numpy as np

from bp_designs.core.geometry import Canvas
from bp_designs.core.scene import Scene
from bp_designs.experiment.params import ParameterSpace, split_params
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.branching.strategies import (
    DefaultAttraction,
    DefaultGrowth,
    ObstacleAvoidanceGrowth,
    VortexAttraction,
)
from bp_designs.generators.primitives.two_d import Oval, Rectangle
from bp_designs.patterns.shape import PointPattern


def generate_pattern(params: dict):
    """Generate pattern from parameters."""
    network_params = split_params(params, "network")
    render_params = split_params(params, "render")

    # Create generator with structural params
    gen = SpaceColonization(**network_params)

    # Generate pattern
    pattern = gen.generate_pattern(**render_params)

    # Wrap in a scene
    canvas = network_params.get("canvas")
    bg_color = render_params.pop("background_color", "#ffffff")

    if canvas:
        canvas.background_color = bg_color
        scene = Scene(canvas)

        # Add obstacles to scene for visualization
        if "growth_strategy" in network_params:
            strategy = network_params["growth_strategy"]
            if isinstance(strategy, ObstacleAvoidanceGrowth):
                for i, obs in enumerate(strategy.shapely_obstacles):
                    # Convert shapely back to our Polygon for rendering
                    from bp_designs.core.geometry import Polygon
                    poly = Polygon(coords=np.array(obs.exterior.coords))
                    scene.add_layer(f"obstacle_{i}", poly.generate_pattern(), fill="#e0e0e0", stroke_color="#999999")

        scene.add_layer("tree", pattern, **render_params)
        return scene

    return pattern


def main():
    """Run advanced strategies exploration."""
    ref_canvas = Canvas.from_size(100)

    # Simple oval boundary
    oval = Oval.from_width_height(900.0, 900.0, canvas=ref_canvas, name="Main boundary")
    boundary = oval.generate_pattern()

    # Create some obstacles
    obs1 = Oval.from_bbox([0.3, 0.3, 0.5, 0.5], canvas=ref_canvas, name="Obstacle 1").generate_pattern().polygon
    obs2 = Rectangle(bbox=(0.6, 0.2, 0.8, 0.4), canvas=ref_canvas, name="Obstacle 2").generate_pattern().polygon
    obstacles = [obs1, obs2]

    root_pos = PointPattern(0.5, 0.95, is_relative=True, name="Bottom Center")

    space = ParameterSpace(
        name="advanced_strategies",
        specs={
            # Network structural parameters
            "network.num_attractions": [400],
            "network.segment_length": [10.0],
            "network.kill_distance": [8.0],
            "network.seed": [42],
            "network.canvas": [ref_canvas],
            "network.initial_boundary": [boundary],
            "network.final_boundary": [boundary],
            "network.root_position": [root_pos],
            # Strategy parameters
            "strategy.mode": ["vortex", "avoidance", "hybrid"],
            # Rendering parameters
            "render.render_mode": "polygon",
            "render.thickness": "descendant",
            "render.min_thickness": 1.0,
            "render.max_thickness": 8.0,
            "render.color_strategy": "depth",
            "render.start_color": "#2c3e50",
            "render.end_color": "#e74c3c",
            "render.background_color": "#ecf0f1",
        },
        derived={
            "network.growth_strategy": lambda p: (
                ObstacleAvoidanceGrowth(obstacles=obstacles, segment_length=p["network.segment_length"])
                if p["strategy.mode"] in ["avoidance", "hybrid"]
                else DefaultGrowth(segment_length=p["network.segment_length"])
            ),
            "network.attraction_strategy": lambda p: (
                VortexAttraction(center=np.array([500.0, 500.0]), strength=2.0)
                if p["strategy.mode"] in ["vortex", "hybrid"]
                else DefaultAttraction()
            )
        }
    )

    grid = space.to_grid()

    runner = ExperimentRunner(
        experiment_name="advanced_strategies",
        svg_width=1000,
        svg_height=1000,
    )

    runner.run(grid=grid, generator_fn=generate_pattern, parallel=False)


if __name__ == "__main__":
    main()
