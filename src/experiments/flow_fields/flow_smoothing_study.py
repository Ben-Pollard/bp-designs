import numpy as np

from bp_designs.core.field import ConstantField, NoiseField
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

    # 1. Field: Base Flow + Noise
    # We use a consistent field to isolate the effects of length and smoothing
    base_field = ConstantField(np.array([1.0, 0.0])) # Right
    noise_field = NoiseField(seed=42, scale=40.0, strength=0.4)
    f = base_field + noise_field

    # 2. Configuration
    config = FlowConfig(
        dt=0.5,
        max_steps=params["max_steps"],
        min_dist=1.0,
        seed_dist=20.0
    )

    # 3. Strategies
    bounds = np.array([[10.0, 10.0], [190.0, 190.0]])
    seeding = GridSeeding(bounds=bounds, resolution=(10, 10))
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

    # 4. Styling (Smoothing via epsilon)
    style = FlowStyle(
        color_strategy=AngleColor(),
        width_strategy=TaperedWidth(min_width=0.2, max_width=1.5),
        epsilon=params["epsilon"]
    )

    return gen.generate_pattern(style=style)

def run_smoothing_study():
    space = ParameterSpace(
        name="flow_smoothing_study",
        specs={
            "max_steps": [50, 150, 300],    # Line Length
            "epsilon": [0.0, 0.5, 2.0, 5.0] # Smoothing (RDP threshold)
        }
    )

    runner = ExperimentRunner("flow_smoothing_study", svg_width=200, svg_height=200)
    runner.run(space.to_grid(), generator_fn, parallel=True)

if __name__ == "__main__":
    run_smoothing_study()
