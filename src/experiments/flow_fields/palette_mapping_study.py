import numpy as np

from bp_designs.core.field import ConstantField, NoiseField, RadialField, VortexField
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
    # High-fidelity canvas
    canvas = Canvas.from_width_height(400, 400)

    # 1. Field Selection
    if params["field_type"] == "vortex":
        # Swirling field centered in canvas
        f = VortexField(center=np.array([200.0, 200.0]), strength=1.0) + NoiseField(
            seed=42, scale=100.0, strength=0.4
        )
    else:
        # Diagonal flow (to avoid the grey 'horizontal' middle-zone of palettes)
        # Pointing towards top-right corner
        base_field = ConstantField(np.array([0.8, 0.8]))
        noise_field = NoiseField(seed=42, scale=150.0, strength=0.6)
        f = base_field + noise_field

    # 2. Configuration
    config = FlowConfig(dt=0.25, max_steps=params["max_steps"], min_dist=2.5, seed_dist=25.0)

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

    # 4. Styling: Range Calibration
    prop = params["map_property"]
    mapping_range = (0.0, 1.0)

    if "position" in prop or "start_x" in prop or "start_y" in prop:
        mapping_range = (0.0, 400.0)
    elif "magnitude" in prop:
        # Tighter range to show noise
        mapping_range = (0.8, 1.2)

    color_strategy = PaletteMapColor(
        palette=params["palette"],
        property=prop,
        range=mapping_range,
        per_segment=params["per_segment"],
        n_bins=32,
    )

    style = FlowStyle(
        color_strategy=color_strategy,
        width_strategy=TaperedWidth(min_width=0.2, max_width=1.5, n_bins=32),
        epsilon=0.1,
    )

    return gen.generate_pattern(style=style)


def run_palette_study():
    space = ParameterSpace(
        name="palette_mapping_study",
        specs={
            "palette": ["UKIYO_E", "MOSS_AND_STONE"],
            "map_property": [
                "angle",  # Directional color
                "position_x",  # Left-to-right fade
                "start_x",  # Solid color by column
                "start_magnitude",  # Solid color by local turbulence
                "magnitude",  # Fading color by local speed
            ],
            "per_segment": [True, False],
            "field_type": ["constant", "vortex"],
            "max_steps": [400],
            "n_bins": [32],
        },
    )

    grid = space.to_grid()

    curated_combinations = []
    for p in grid.combinations:
        prop = p["map_property"]
        ps = p["per_segment"]

        # Select best illustrative variants
        is_relevant = False
        if ps:  # Fading
            if prop in ["angle", "position_x", "magnitude"]:
                is_relevant = True
        else:  # Solid
            if prop in ["start_x", "angle", "start_magnitude"]:
                is_relevant = True

        if not is_relevant:
            continue

        # Prioritize UKIYO_E for vibrant red terracotta
        if p["palette"] == "UKIYO_E":
            curated_combinations.append(p)
        elif (
            p["palette"] == "MOSS_AND_STONE"
            and p["map_property"] == "angle"
            and p["field_type"] == "vortex"
            and p["per_segment"]
        ):
            # One fading angle example for Moss
            curated_combinations.append(p)

    from bp_designs.experiment.params import ParameterGrid

    final_grid = ParameterGrid(
        space_name=grid.space_name,
        param_names=grid.param_names,
        varied_params=grid.varied_params,
        fixed_params=grid.fixed_params,
        combinations=curated_combinations,
    )

    runner = ExperimentRunner("palette_mapping_study", svg_width=400, svg_height=400)
    runner.run(final_grid, generator_fn, parallel=True)


if __name__ == "__main__":
    run_palette_study()
