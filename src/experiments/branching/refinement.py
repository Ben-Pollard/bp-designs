"""Experiment demonstrating blossoms on branching networks."""

from bp_designs.core.geometry import Canvas
from bp_designs.experiment.params import ParameterSpace, split_params
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.primitives.two_d import Oval
from bp_designs.patterns.network.refinement import NetworkRefinementStrategy
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
    ref_canvas = Canvas.from_size(200)#.with_background_color("#d2dbaf")

    # Simple oval boundary
    oval = Oval.from_width_height(0.75, 0.75, canvas=ref_canvas, name="Circular boundary")
    boundary = oval.generate_pattern()

    root_pos = PointPattern(0.5, 0.5, is_relative=True, name="Center")


    space = ParameterSpace(
        name="refinement",
        specs={
            # Network structural parameters
            "network.num_attractions": [100],
            "network.segment_length": [2.5],
            "network.kill_distance": [2],
            "network.organ_distribution": ["terminal"],
            "network.seed": [42],
            "network.canvas": [ref_canvas],
            "network.max_iterations": [100],
            "network.initial_boundary": [boundary],
            "network.final_boundary": [boundary],
            "network.root_position": [root_pos],
            "refinement.dist": [None, 3],
            "refinement.subdivide": [True],
            "refinement.alpha": [0.1, 0.25],
            "refinement.it": [3, 10],
            # "network.organ_template": [blossom_template],
            # Rendering parameters
            "render.render_mode": "polyline",
            "render.taper_style": "smooth",
            "render.shading": "linear",
            "render.color_strategy": "depth",
            "render.thickness": "descendant",
            "render.min_thickness": 0.5,
            "render.max_thickness": [5],
            # "render.background_color": "#f0f0e0",
            "render.start_color": "#3d2b1f",
            "render.end_color": "#5d4037",
            # "render.background_color": "#fdf5e6",
            # Lighting parameters
            "render.light_angle": [45],
            "render.highlight_amount": [0.2],
            "render.shadow_amount": [0.2],
            "render.bg_saturation": [0.5],
            "render.bg_lightness": [0.94]
        },
        derived={
            # # Create the LightingModel object from parameters
            # "render.background_color": lambda p: p["network.organ_template"]
            # .base_color.complementary()
            # .with_hsl(s=p["render.bg_saturation"], lightness=p["render.bg_lightness"]),
            # "render.lighting": lambda p: DirectionalLighting(
            #     light_direction=np.array([
            #         np.cos(np.radians(p["render.light_angle"])),
            #         np.sin(np.radians(p["render.light_angle"]))
            #     ]),
            #     highlight_amount=p["render.highlight_amount"],
            #     shadow_amount=p["render.shadow_amount"]
            # )
            "network.refinement_strategy": lambda p: NetworkRefinementStrategy(
        decimate_min_distance=p['refinement.dist'], subdivide=p['refinement.subdivide'], relocate_alpha=p['refinement.alpha'], relocate_iterations=p['refinement.it'])
        },
    )

    grid = space.to_grid()

    runner = ExperimentRunner(
        experiment_name="refinement",
        svg_width=1000,
        svg_height=1000,
    )

    runner.run(grid=grid, generator_fn=generate_pattern, parallel=True)


if __name__ == "__main__":
    main()
