"""Experiment demonstrating Crystalline (grid-snapped) Space Colonization."""


from bp_designs.core.geometry import Canvas
from bp_designs.core.scene import Scene
from bp_designs.experiment.params import ParameterSpace, split_params
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.branching.strategies import GridSnappedGrowth
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
    """Run crystalline growth exploration."""
    ref_canvas = Canvas.from_size(1000)

    # Simple oval boundary
    oval = Oval.from_width_height(800.0, 800.0, canvas=ref_canvas, name="Circular boundary")
    boundary = oval.generate_pattern()

    root_pos = PointPattern(0.5, 1.0, is_relative=True, name="Bottom Center")

    space = ParameterSpace(
        name="crystalline_growth",
        specs={
            # Network structural parameters
            "network.num_attractions": [500],
            "network.segment_length": [10.0],
            "network.kill_distance": [8.0],
            "network.seed": [42, 123],
            "network.canvas": [ref_canvas],
            "network.initial_boundary": [boundary],
            "network.final_boundary": [boundary],
            "network.root_position": [root_pos],
            # Strategy parameters
            "strategy.angles": [4, 8],
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
            "network.growth_strategy": lambda p: GridSnappedGrowth(
                segment_length=p["network.segment_length"],
                angles=p["strategy.angles"]
            )
        }
    )

    grid = space.to_grid()

    runner = ExperimentRunner(
        experiment_name="crystalline_growth",
        svg_width=1000,
        svg_height=1000,
    )

    runner.run(grid=grid, generator_fn=generate_pattern, parallel=False)


if __name__ == "__main__":
    main()
