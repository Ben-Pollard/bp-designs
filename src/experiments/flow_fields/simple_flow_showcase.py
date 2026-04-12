import numpy as np

from bp_designs.core.field import ConstantField, NoiseField
from bp_designs.core.geometry import Canvas
from bp_designs.experiment.params import ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.flow.classic import ClassicFlowGenerator
from bp_designs.generators.flow.grid import GridFieldGenerator
from bp_designs.generators.flow.strategies import (
    GridSeeding,
    RandomSeeding,
)
from bp_designs.patterns.flow import (
    FlowStyle,
    PaletteMapColor,
    TaperedWidth,
)


def generator_fn(params):
    canvas = Canvas.from_width_height(200, 200)

    # 1. Field: Constant + Noise
    f = ConstantField(np.array([1.0, 0.2])) + NoiseField(
        seed=params["seed"], scale=params["noise_scale"], strength=params["noise_strength"]
    )

    # 2. Generator selection
    if params["generator_type"] == "classic":
        seeding = RandomSeeding(
            bounds=np.array([[10, 10], [190, 190]]),
            num_seeds=params["num_seeds"],
            seed=params["seed"],
        )
        gen = ClassicFlowGenerator(
            canvas=canvas,
            field=f,
            seeding_strategy=seeding,
            max_steps=params["max_steps"],
            dt=0.5,
            min_dist=params["min_dist"],
            trace_both_ways=True,
        )
    else:  # "grid"
        seeding = GridSeeding(bounds=np.array([[10, 10], [190, 190]]), resolution=(20, 20))
        gen = GridFieldGenerator(
            canvas=canvas, field=f, seeding_strategy=seeding, length=params["grid_length"]
        )

    style = FlowStyle(
        color_strategy=PaletteMapColor(palette="MOSS_AND_STONE", property="angle"),
        width_strategy=TaperedWidth(min_width=0.2, max_width=1.5),
        epsilon=0.1,
    )

    return gen.generate_pattern(style=style)


def run_experiment():
    space = ParameterSpace(
        name="simple_flow_showcase",
        specs={
            "generator_type": ["classic", "grid"],
            "noise_strength": [0.8],
            "noise_scale": [100.0],
            "num_seeds": [300],
            "max_steps": [500],
            "min_dist": [0.0, 2.0],
            "grid_length": [10.0],
            "seed": [42],
        },
    )

    runner = ExperimentRunner("simple_flow_showcase", svg_width=300, svg_height=300)
    runner.run(space.to_grid(), generator_fn, parallel=True)


if __name__ == "__main__":
    run_experiment()
