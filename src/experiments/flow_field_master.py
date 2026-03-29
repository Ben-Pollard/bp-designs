import numpy as np

from bp_designs.core.field import NoiseField, RadialField
from bp_designs.core.geometry import Canvas
from bp_designs.experiment.params import ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.flow.generator import FlowGenerator
from bp_designs.generators.flow.strategies import (
    FixedLengthTermination,
    PoissonDiscSeeding,
    RK4Integrator,
)


def generator_fn(params):
    canvas = Canvas.from_width_height(200, 200)
    radial_field = RadialField(center=np.array([100.0, 100.0]), strength=0.5)

    # Update field with params if needed
    f = NoiseField(seed=params["seed"], scale=params["scale"], strength=1.0) + radial_field

    seeding = PoissonDiscSeeding(
        bounds=np.array([[10.0, 10.0], [190.0, 190.0]]),
        min_dist=10.0,
        seed=42
    )

    integrator = RK4Integrator()
    termination = FixedLengthTermination(max_steps=100)

    gen = FlowGenerator(
        canvas=canvas,
        field=f,
        seeding_strategy=seeding,
        integration_strategy=integrator,
        termination_strategy=termination,
        dt=params["dt"]
    )
    return gen.generate_pattern(
        taper=True,
        color_mode="angle",
        epsilon=0.1
    )

def run_flow_experiment():
    # 3. Run experiment
    space = ParameterSpace(
        name="flow_master",
        specs={
            "seed": [42, 123],
            "scale": [30.0, 60.0],
            "dt": 0.5
        }
    )

    runner = ExperimentRunner("flow_field_master", svg_width=200, svg_height=200)
    runner.run(space.to_grid(), generator_fn, parallel=False)

if __name__ == "__main__":
    run_flow_experiment()
