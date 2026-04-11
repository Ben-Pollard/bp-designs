import numpy as np

from bp_designs.core.field import (
    ConstantField,
    NoiseField,
    SineWaveField,
    ValueNoiseField,
    WaveletField,
    WorleyField,
)
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

    # 1. Select Noise Engine
    engine = params["engine"]
    if engine == "simplex":
        noise = NoiseField(seed=42, scale=40.0, strength=1.0)
    elif engine == "worley":
        noise = WorleyField(seed=42, num_points=15, strength=1.0)
    elif engine == "value":
        noise = ValueNoiseField(seed=42, grid_res=8, strength=1.0)
    elif engine == "sine":
        noise = SineWaveField(seed=42, num_waves=4, scale=30.0, strength=1.0)
    elif engine == "wavelet":
        noise = WaveletField(seed=42, scale=30.0, strength=1.0)
    else:
        noise = NoiseField(seed=42)

    # 2. Compose with a base flow to see how it "organicizes" structure
    base_flow = ConstantField(np.array([1.0, 0.0])) # Right
    f = base_flow + noise * params["noise_weight"]

    # 3. Configuration
    config = FlowConfig(
        dt=0.5,
        max_steps=200,
        min_dist=1.0,
        seed_dist=15.0
    )

    # 4. Strategies
    bounds = np.array([[10.0, 10.0], [190.0, 190.0]])
    seeding = GridSeeding(bounds=bounds, resolution=(12, 12))
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

def run_engine_comparison():
    space = ParameterSpace(
        name="noise_engine_comparison",
        specs={
            "engine": ["simplex", "worley", "value", "sine", "wavelet"],
            "noise_weight": [0.3, 0.7] # Low vs High perturbation
        }
    )

    runner = ExperimentRunner("noise_engine_comparison", svg_width=200, svg_height=200)
    runner.run(space.to_grid(), generator_fn, parallel=True)

if __name__ == "__main__":
    run_engine_comparison()
