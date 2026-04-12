import numpy as np

from bp_designs.core.field import (
    AngleField,
    ConstantField,
    NoiseField,
    RadialField,
    SineWaveField,
    ValueNoiseField,
    VortexField,
    WaveletField,
    WorleyField,
)
from bp_designs.core.geometry import Canvas
from bp_designs.experiment.params import ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.flow.generator import FlowConfig, FlowGenerator
from bp_designs.generators.flow.classic import ClassicFlowGenerator
from bp_designs.generators.flow.strategies import (
    AngleJoinStrategy,
    CompositeTermination,
    EulerIntegrator,
    FixedLengthTermination,
    GridSeeding,
    PoissonDiscSeeding,
    ProximityTermination,
    RandomSeeding,
    RK4Integrator,
)
from bp_designs.patterns.flow import (
    AngleColor,
    ConstantColor,
    FlowStyle,
    MagnitudeColor,
    MagnitudeWidth,
    PaletteMapColor,
    TaperedWidth,
)


def generator_fn(params):
    canvas = Canvas.from_width_height(200, 200)
    center = np.array([100.0, 100.0])

    # 1. Field Composition
    base_type = params["base_field"]
    if base_type == "constant":
        base_field = ConstantField(np.array([1.0, 0.0]))  # Right
    elif base_type == "angle":
        base_field = AngleField(params["angle"], degrees=True)
    elif base_type == "radial":
        base_field = RadialField(center=center, strength=1.0)
    elif base_type == "vortex":
        base_field = VortexField(center=center, strength=1.0, clockwise=True)
    else:
        base_field = ConstantField(np.array([0.0, 0.0]))

    engine = params["engine"]
    if engine == "simplex":
        noise_field = NoiseField(seed=42, scale=params["noise_scale"], strength=params["noise_strength"])
    elif engine == "worley":
        noise_field = WorleyField(seed=42, num_points=15, strength=params["noise_strength"])
    elif engine == "value":
        noise_field = ValueNoiseField(seed=42, grid_res=8, strength=params["noise_strength"])
    elif engine == "sine":
        noise_field = SineWaveField(seed=42, num_waves=4, scale=30.0, strength=params["noise_strength"])
    elif engine == "wavelet":
        noise_field = WaveletField(seed=42, scale=30.0, strength=params["noise_strength"])
    else:
        noise_field = NoiseField(seed=42)

    f = base_field + noise_field

    join_strategy = None
    if params["join_enabled"]:
        join_strategy = AngleJoinStrategy(
            max_angle_deg=params["join_angle"], endpoint_only=params["endpoint_only"]
        )

    # 2. Structural Configuration (Explicit)
    config = FlowConfig(
        dt=params["dt"],
        max_steps=params["max_steps"],
        min_dist=params["min_dist"],
        seed_dist=params["seed_dist"],
        steering_radius=params["steering_radius"],
        steering_strength=params["steering_strength"],
        steering_lookahead=params["steering_lookahead"],
    )

    # 3. Strategies
    bounds = np.array([[10.0, 10.0], [190.0, 190.0]])

    if params["seeding"] == "poisson":
        seeding = PoissonDiscSeeding(bounds=bounds, min_dist=config.seed_dist, seed=42)
    elif params["seeding"] == "grid":
        seeding = GridSeeding(bounds=bounds, resolution=(10, 10))
    elif params["seeding"] == "random":
        seeding = RandomSeeding(bounds=bounds, num_seeds=int(canvas.height * canvas.width / config.seed_dist))

    # Combine FixedLength and Proximity using the core CompositeTermination
    termination = CompositeTermination(
        [
            FixedLengthTermination(max_steps=config.max_steps),
            ProximityTermination(existing_points=np.empty((0, 2)), min_dist=config.min_dist),
        ]
    )
    integrator = RK4Integrator() if params["integrator"] == "rk4" else EulerIntegrator()

    gen = ClassicFlowGenerator(
        canvas=canvas,
        field=f,
        seeding_strategy=seeding,
        integration_strategy=integrator,
        config=config,
        trace_both_ways=True,
    )

    # 4. Rendering Style (Explicit FlowStyle object)
    color_mode = params["color_mode"]
    if color_mode == "angle":
        color_strategy = AngleColor()
    elif color_mode == "magnitude":
        color_strategy = MagnitudeColor(color1="#440154", color2="#fde725", mag_range=(0.0, 1.0))
    elif color_mode == "palette":
        color_strategy = PaletteMapColor(
            palette=params["palette"], property=params["map_property"], interpolate=True, per_segment=False
        )
    else:
        color_strategy = ConstantColor(color="#000000")

    width_mode = params["width_mode"]
    if width_mode == "taper":
        width_strategy = TaperedWidth(min_width=0.2, max_width=2.0)
    elif width_mode == "magnitude":
        width_strategy = MagnitudeWidth(min_width=0.2, max_width=3.0, mag_range=(0.0, 1.0))
    else:
        width_strategy = None  # Uses defaults in FlowStyle

    style = FlowStyle(color_strategy=color_strategy, width_strategy=width_strategy, epsilon=params["epsilon"])

    return gen.generate_pattern(style=style)


def run_density_experiment():
    space = ParameterSpace(
        name="flow_density_study",
        specs={
            "base_field": ["constant"],
            "integrator": ["rk4"],
            "seeding": ["random", "grid"],
            "dt": [0.5],
            "noise_strength": [0.9],  # Balance between chaos and order
            "noise_scale": [50, 100],  # Frequency of noise
            "seed_dist": [250, 1000],  # How many lines start
            "min_dist": [10],  # How close they can get (avoidance)
            "max_steps": [1000],  # Max line length
            "color_mode": ["palette"],
            "width_mode": ["taper"],
            "epsilon": [0],  # Simplification
            "engine": ["simplex"],  # ["worley", "value", "sine", "wavelet"]
            "palette": ["MOSS_AND_STONE"],  # ["UKIYO_E", "BOTANICAL"],
            "map_property": [
                "angle"
            ],  # ["position_x", "position_y", "start_x", "start_y", "start_magnitude"]
            "steering_radius": [50, 200],
            "steering_strength": [1, 2, 10],
            "steering_lookahead": [50, 200],
            "join_enabled": [True],
            "join_angle": [0, 10],
            "endpoint_only": [True],
        },
    )

    runner = ExperimentRunner("flow_monochrome", svg_width=200, svg_height=200)
    runner.run(space.to_grid(), generator_fn, parallel=True)


if __name__ == "__main__":
    run_density_experiment()
