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
from bp_designs.patterns.flow import (
    FlowStyle,
    PaletteMapColor,
    TaperedWidth,
)


def generator_fn(params):
    canvas = Canvas.from_width_height(400, 400)

    # 1. Field: Base Flow + Noise
    base_field = ConstantField(np.array([1.0, 0.0]))  # Right
    noise_field = NoiseField(seed=42, scale=100.0, strength=0.8)
    f = base_field + noise_field

    # 2. Configuration
    config = FlowConfig(dt=0.2, max_steps=600, min_dist=2.0, seed_dist=15.0)

    # 3. Strategies
    bounds = np.array([[10.0, 10.0], [390.0, 390.0]])
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

    # 4. Styling: Palette Mapping
    # Map the chosen property to a Master Palette
    # Set range based on property
    mapping_range = (0.0, 1.0)
    if "position" in params["map_property"] or "start" in params["map_property"]:
        mapping_range = (0.0, 400.0)
    elif "magnitude" in params["map_property"]:
        mapping_range = (0.0, 1.5)  # Approximate magnitude range for Constant(1.0) + Noise(0.8)

    color_strategy = PaletteMapColor(
        palette=params["palette"], property=params["map_property"], range=mapping_range
    )

    style = FlowStyle(
        color_strategy=color_strategy,
        width_strategy=TaperedWidth(min_width=0.2, max_width=1.5),
        epsilon=0.2,  # Lower epsilon for smoother visual but some simplification
    )

    return gen.generate_pattern(style=style)


def run_palette_study():
    space = ParameterSpace(
        name="palette_mapping_study",
        specs={
            "palette": ["MOSS_AND_STONE", "UKIYO_E", "BOTANICAL"],
            "map_property": ["angle", "position_x", "position_y", "start_x", "start_y", "start_magnitude"],
        },
    )

    runner = ExperimentRunner("palette_mapping_study", svg_width=200, svg_height=200)
    runner.run(space.to_grid(), generator_fn, parallel=True)


if __name__ == "__main__":
    run_palette_study()
