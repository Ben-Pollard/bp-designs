import numpy as np

from bp_designs.core.field import NoiseField, RadialField
from bp_designs.core.geometry import Canvas
from bp_designs.experiment.params import ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.flow.generator import FlowConfig, FlowGenerator
from bp_designs.generators.flow.strategies import (
    EulerIntegrator,
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
    TaperedWidth,
)


def generator_fn(params):
    canvas = Canvas.from_width_height(200, 200)

    # 1. Field Composition: Balance Noise vs Radial
    noise_weight = params["noise_weight"]
    radial_weight = 1.0 - noise_weight

    radial_field = RadialField(center=np.array([100.0, 100.0]), strength=radial_weight)
    noise_field = NoiseField(seed=42, scale=params["noise_scale"], strength=noise_weight)

    f = noise_field + radial_field

    # 2. Structural Configuration (Explicit)
    config = FlowConfig(
        dt=params["dt"],
        max_steps=params["max_steps"],
        min_dist=params["min_dist"],
        seed_dist=params["seed_dist"]
    )

    # 3. Strategies
    bounds = np.array([[10.0, 10.0], [190.0, 190.0]])

    if params["seeding"] == "poisson":
        seeding = PoissonDiscSeeding(
            bounds=bounds,
            min_dist=config.seed_dist,
            seed=42
        )
    elif params["seeding"] == "grid":
        seeding = GridSeeding(bounds=bounds, resolution=(10,10))
    elif params["seeding"] == "random":
        seeding = RandomSeeding(bounds=bounds, num_seeds=int(canvas.height * canvas.width / config.seed_dist ))

    # Combine FixedLength and Proximity using a simple wrapper
    class CompositeTermination:
        def __init__(self, max_steps, min_dist):
            self.max_steps = max_steps
            self.proximity = ProximityTermination(existing_points=np.empty((0, 2)), min_dist=min_dist)

        def should_terminate(self, pos, step, ids=None):
            # Stop if too many steps OR if too close to others
            too_long = step >= self.max_steps
            too_close = self.proximity.should_terminate(pos, step, ids=ids)
            return too_long | too_close

        def update(self, pos, ids=None):
            self.proximity.update(pos, ids=ids)

    termination = CompositeTermination(config.max_steps, config.min_dist)
    integrator = RK4Integrator() if params["integrator"] == "rk4" else EulerIntegrator()

    gen = FlowGenerator(
        canvas=canvas,
        field=f,
        seeding_strategy=seeding,
        integration_strategy=integrator,
        termination_strategy=termination,
        config=config
    )

    # 4. Rendering Style (Explicit FlowStyle object)
    color_mode = params["color_mode"]
    if color_mode == "angle":
        color_strategy = AngleColor()
    elif color_mode == "magnitude":
        color_strategy = MagnitudeColor(color1="#440154", color2="#fde725", mag_range=(0.0, 1.0))
    else:
        color_strategy = ConstantColor(color="#000000")

    width_mode = params["width_mode"]
    if width_mode == "taper":
        width_strategy = TaperedWidth(min_width=0.2, max_width=2.0)
    elif width_mode == "magnitude":
        width_strategy = MagnitudeWidth(min_width=0.2, max_width=3.0, mag_range=(0.0, 1.0))
    else:
        width_strategy = None # Uses defaults in FlowStyle

    style = FlowStyle(
        color_strategy=color_strategy,
        width_strategy=width_strategy,
        epsilon=params["epsilon"]
    )

    return gen.generate_pattern(style=style)

def run_density_experiment():
    space = ParameterSpace(
        name="flow_density_study",
        specs={
            "integrator": ["rk4"],
            "seeding": ["random", "poisson"],
            "dt": [0.5],
            "noise_weight": [0.9],      # Balance between chaos and order
            "noise_scale": [50, 100],     # Frequency of noise
            "seed_dist": [10, 100],             # How many lines start
            "min_dist": [1.0],          # How close they can get (avoidance)
            "max_steps": [500],         # Max line length
            "color_mode": ["angle"],
            "width_mode": ["taper"],
            "epsilon": [0.5]           # Simplification
        }
    )

    runner = ExperimentRunner("flow_density_study", svg_width=200, svg_height=200)
    runner.run(space.to_grid(), generator_fn, parallel=True)

if __name__ == "__main__":
    run_density_experiment()
