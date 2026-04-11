"""Experiment demonstrating various Space Colonization growth strategies."""


from bp_designs.core.geometry import Canvas
from bp_designs.core.scene import Scene
from bp_designs.experiment.params import ParameterSpace, split_params
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.branching.strategies import DefaultGrowth, GridSnappedGrowth, MomentumGrowth
from bp_designs.generators.primitives.two_d import Oval
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
        scene.add_layer("tree", pattern, **render_params)
        return scene

    return pattern


def main():
    """Run growth strategies exploration."""
    ref_canvas = Canvas.from_size(1000)

    # Simple oval boundary
    oval = Oval.from_width_height(800.0, 800.0, canvas=ref_canvas, name="Circular boundary")
    boundary = oval.generate_pattern()

    root_pos = PointPattern(0.5, 1.0, is_relative=True, name="Bottom Center")

    space = ParameterSpace(
        name="growth_strategies",
        specs={
            # Network structural parameters
            "network.num_attractions": [20],
            "network.segment_length": [10.0],
            "network.kill_distance": [8.0],
            "network.seed": [42],
            "network.canvas": [ref_canvas],
            "network.initial_boundary": [boundary],
            "network.final_boundary": [boundary],
            "network.root_position": [root_pos],
            # Strategy parameters
            "strategy.type": ["default", "momentum", "grid_4", "grid_8"],
            "strategy.momentum_value": [0.1, 0.3, 0.5],
            # Rendering parameters
            "render.render_mode": "polygon",
            "render.thickness": "descendant",
            "render.min_thickness": 1.0,
            "render.max_thickness": 10.0,
            "render.color_strategy": "depth",
            "render.start_color": "#1a1a1a",
            "render.end_color": "#444444",
            "render.background_color": "#f8f8f8",
        },
        derived={
            "network.growth_strategy": lambda p: (
                MomentumGrowth(segment_length=p["network.segment_length"], momentum=p["strategy.momentum_value"])
                if p["strategy.type"] == "momentum"
                else GridSnappedGrowth(segment_length=p["network.segment_length"], angles=4)
                if p["strategy.type"] == "grid_4"
                else GridSnappedGrowth(segment_length=p["network.segment_length"], angles=8)
                if p["strategy.type"] == "grid_8"
                else DefaultGrowth(segment_length=p["network.segment_length"])
            )
        }
    )

    grid = space.to_grid()

    runner = ExperimentRunner(
        experiment_name="growth_strategies",
        svg_width=1000,
        svg_height=1000,
    )

    runner.run(grid=grid, generator_fn=generate_pattern, parallel=True)


if __name__ == "__main__":
    main()
