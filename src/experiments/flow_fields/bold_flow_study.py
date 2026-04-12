import numpy as np

from bp_designs.core.field import ConstantField, NoiseField
from bp_designs.core.geometry import Canvas
from bp_designs.experiment.params import ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.flow.classic import ClassicFlowGenerator
from bp_designs.generators.flow.generator import FlowConfig
from bp_designs.generators.flow.strategies import (
    GridSeeding,
    RK4Integrator,
)
from bp_designs.patterns.flow import ConstantWidth, FlowStyle, PaletteMapColor, TaperedWidth


def generator_fn(params):
    canvas = Canvas.from_width_height(200, 200)

    # 1. Field: Base Flow + Noise
    base_field = ConstantField(np.array([1.0, 0.0]))  # Right
    noise_field = NoiseField(seed=42, scale=50.0, strength=0.5)
    f = base_field + noise_field

    # 2. Configuration
    config = FlowConfig(dt=0.2, max_steps=500, min_dist=params["min_dist"], seed_dist=15.0)

    # 3. Strategies
    bounds = np.array([[10.0, 10.0], [190.0, 190.0]])
    seeding = GridSeeding(bounds=bounds, resolution=(15, 15))
    integrator = RK4Integrator()

    gen = ClassicFlowGenerator(
        canvas=canvas,
        field=f,
        seeding_strategy=seeding,
        integration_strategy=integrator,
        config=config,
        trace_both_ways=True,
    )

    # 4. Styling: Bold vs Organic
    color_strategy = PaletteMapColor(
        palette="UKIYO_E",
        property="angle",
        interpolate=params["interpolate"],
        per_segment=params["per_segment"],
    )

    if params["width_mode"] == "constant":
        width_strategy = ConstantWidth(width=params["max_width"])
    else:
        width_strategy = TaperedWidth(min_width=0.2, max_width=params["max_width"])

    style = FlowStyle(color_strategy=color_strategy, width_strategy=width_strategy, epsilon=0.5)

    return gen.generate_pattern(style=style)


def run_bold_study():
    space = ParameterSpace(
        name="bold_flow_study",
        specs={
            "min_dist": [0.0, 1.5, 4.0],  # Spacing
            "max_width": [1.0, 3.0, 6.0],  # Thickness
            "width_mode": ["constant", "tapered"],
            "interpolate": [True],
            "per_segment": [True],
        },
    )

    runner = ExperimentRunner("bold_flow_study", svg_width=200, svg_height=200)
    runner.run(space.to_grid(), generator_fn, parallel=True)


if __name__ == "__main__":
    run_bold_study()
