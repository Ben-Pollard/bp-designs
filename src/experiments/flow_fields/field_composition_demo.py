import numpy as np

from bp_designs.core.field import AngleField, ConstantField, NoiseField, RadialField, VortexField
from bp_designs.core.geometry import Canvas
from bp_designs.experiment.params import ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.flow.generator import FlowConfig, FlowGenerator
from bp_designs.generators.flow.strategies import (
    FixedLengthTermination,
    GridSeeding,
    RK4Integrator,
)
from bp_designs.patterns.flow import (
    AngleColor,
    FlowStyle,
    TaperedWidth,
)


def generator_fn(params):
    canvas = Canvas.from_width_height(200, 200)
    center = np.array([100.0, 100.0])

    # 1. Select Base Field
    base_type = params["base_field"]
    if base_type == "constant":
        base_field = ConstantField(np.array([1.0, 0.0])) # Right
    elif base_type == "angle":
        base_field = AngleField(params["angle"], degrees=True)
    elif base_type == "radial":
        base_field = RadialField(center=center, strength=1.0)
    elif base_type == "vortex":
        base_field = VortexField(center=center, strength=1.0, clockwise=True)
    else:
        base_field = ConstantField(np.array([0.0, 0.0]))

    # 2. Compose with Noise
    noise_field = NoiseField(seed=42, scale=params["noise_scale"], strength=params["noise_strength"])

    # The "Joined Up" effect comes from adding the base intent to the noise
    f = base_field + noise_field

    # 3. Configuration
    config = FlowConfig(
        dt=0.5,
        max_steps=params["max_steps"],
        min_dist=1.0,
        seed_dist=params["grid_res"]
    )

    # 4. Strategies
    bounds = np.array([[10.0, 10.0], [190.0, 190.0]])
    # Grid seeding makes the underlying structure/distortion very clear
    res = int(180 / params["grid_res"])
    seeding = GridSeeding(bounds=bounds, resolution=(res, res))

    integrator = RK4Integrator()
    termination = FixedLengthTermination(max_steps=config.max_steps)

    gen = FlowGenerator(
        canvas=canvas,
        field=f,
        seeding_strategy=seeding,
        integration_strategy=integrator,
        termination_strategy=termination,
        config=config
    )

    # 5. Styling
    style = FlowStyle(
        color_strategy=AngleColor(),
        width_strategy=TaperedWidth(min_width=0.2, max_width=1.5),
        epsilon=0.5
    )

    return gen.generate_pattern(style=style)

def run_composition_demo():
    space = ParameterSpace(
        name="field_composition_demo",
        specs={
            "base_field": ["constant", "angle", "radial", "vortex"],
            "angle": [45],
            "noise_strength": [0.1, 0.3, 0.6], # Transition from "Joined Up" to "Chaotic"
            "noise_scale": [50],
            "grid_res": [15],
            "max_steps": [200]
        }
    )

    runner = ExperimentRunner("field_composition_demo", svg_width=200, svg_height=200)
    runner.run(space.to_grid(), generator_fn, parallel=True)

if __name__ == "__main__":
    run_composition_demo()
