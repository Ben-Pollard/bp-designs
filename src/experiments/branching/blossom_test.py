"""Experiment to visualize and test the new BlossomOrganPattern."""

import numpy as np
import svgwrite

from bp_designs.core.lighting import DirectionalLighting
from bp_designs.core.renderer import RenderingContext, RenderStyle
from bp_designs.experiment.params import ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.patterns.organs import BlossomGenerator


def generate_blossom(params: dict):
    """Generate a blossom pattern for preview."""
    # This is a bit different from network experiments as we just want to render the organ
    # We'll wrap it in a dummy object that has a to_svg method

    blossom_gen = BlossomGenerator()
    pattern = blossom_gen.generate_pattern(**params)

    class PreviewPattern:
        def __init__(self, p):
            self.p = p
            self.name = p.name

        def to_svg(self, **kwargs) -> str:
            # Use a larger viewBox for blossoms
            dwg = svgwrite.Drawing(size=("400px", "400px"), viewBox="-40 -40 80 80")

            # Add background
            bg_color = kwargs.get("background_color", "#f0f0f0")
            dwg.add(dwg.rect(insert=(-40, -40), size=(80, 80), fill=bg_color))

            # Setup lighting if provided in params
            lighting = params.get("lighting")
            context = RenderingContext(dwg, lighting=lighting)

            style = RenderStyle()
            self.p.render(context, np.array([0.0, 0.0]), style=style)
            return dwg.tostring()

    return PreviewPattern(pattern)


def main():
    """Run blossom exploration."""

    space = ParameterSpace(
        name="blossom_exploration",
        specs={
            "num_rings": [2, 3, 4],
            "base_petal_count": [5],
            "petal_shape": ["teardrop", "heart"],
            "base_color": ["#ff69b4", "#ffcc00", "#ffffff"],
            "jitter": [0.1],
            "overlap": [1.2],
            "inner_radius": [0.0, 1.0],
            "ring_spacing": [1.0, 2.0],
            "scale": [3.0],
            "light_angle": [45],
        },
        derived={
            "lighting": lambda p: DirectionalLighting(
                light_direction=np.array([
                    np.cos(np.radians(p["light_angle"])),
                    np.sin(np.radians(p["light_angle"]))
                ]),
                highlight_amount=0.2,
                shadow_amount=0.2
            )
        }
    )

    grid = space.to_grid()

    runner = ExperimentRunner(
        experiment_name="blossom_variations",
        svg_width=400,
        svg_height=400,
    )

    runner.run(grid=grid, generator_fn=generate_blossom)


if __name__ == "__main__":
    main()
