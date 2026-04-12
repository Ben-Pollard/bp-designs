import numpy as np

from bp_designs.core.field import ConstantField, NoiseField
from bp_designs.core.geometry import Canvas
from bp_designs.experiment.params import ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.flow.generator import FlowConfig, FlowGenerator
from bp_designs.generators.flow.strategies import (
    AngleJoinStrategy,
    PoissonDiscSeeding,
    ProximityTermination,
    RK4Integrator,
)
from bp_designs.patterns.flow import (
    FlowStyle,
    PaletteMapColor,
    TaperedWidth,
)


def generator_fn(params):
    canvas = Canvas.from_width_height(200, 200)

    # Field: Constant flow with some noise to create convergence
    f = ConstantField(np.array([1.0, 0.2])) + NoiseField(seed=42, scale=100.0, strength=0.5)

    # Join Strategy
    join_strategy = None
    if params["join_enabled"]:
        join_strategy = AngleJoinStrategy(
            max_angle_deg=params["join_angle"], endpoint_only=params["endpoint_only"]
        )

    config = FlowConfig(
        dt=0.1,  # High resolution
        max_steps=params["max_steps"],
        min_dist=params["min_dist"],
        seed_dist=params["seed_dist"],
        join_strategy=join_strategy,
        steering_radius=30.0,
        steering_strength=2.0,
        steering_lookahead=params["steering_lookahead"],
    )

    bounds = np.array([[10.0, 10.0], [190.0, 190.0]])
    seeding = PoissonDiscSeeding(bounds=bounds, min_dist=config.seed_dist, seed=42)

    # Use ProximityTermination directly to support joining
    termination = ProximityTermination(existing_points=np.empty((0, 2)), min_dist=config.min_dist)
    integrator = RK4Integrator()

    gen = FlowGenerator(
        canvas=canvas,
        field=f,
        seeding_strategy=seeding,
        integration_strategy=integrator,
        termination_strategy=termination,
        config=config,
    )

    style = FlowStyle(
        color_strategy=PaletteMapColor(palette="MOSS_AND_STONE", property="angle"),
        width_strategy=TaperedWidth(min_width=0.2, max_width=2.0),
        epsilon=0.0,  # RAW GEOMETRY
    )

    return gen.generate_pattern(style=style)


def run_join_experiment():
    space = ParameterSpace(
        name="flow_join_study",
        specs={
            "join_enabled": [True],
            "join_angle": [15, 30],
            "steering_lookahead": [10.0, 40.0],
            "endpoint_only": [True],
            "seed_dist": [15.0],
            "min_dist": [1.5],
            "max_steps": [500],
        },
    )

    runner = ExperimentRunner("flow_join_study", svg_width=400, svg_height=400)
    runner.run(space.to_grid(), generator_fn, parallel=True)


if __name__ == "__main__":
    run_join_experiment()
