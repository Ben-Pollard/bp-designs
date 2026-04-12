from __future__ import annotations

import numpy as np
from bp_designs.core.generator import Generator
from bp_designs.core.geometry import Canvas
from bp_designs.patterns.flow import StreamlinePattern
from bp_designs.generators.flow.strategies import SeedingStrategy


class GridFieldGenerator(Generator):
    """A grid-based flow field generator that doesn't use particle tracing.

    Instead, it just draws a line segment at each seed point pointing in the
    direction of the field at that point. This is the simplest 'flow field'
    implementation often seen in tutorials.
    """

    def __init__(
        self,
        canvas: Canvas,
        field,
        seeding_strategy: SeedingStrategy,
        length: float = 10.0,
    ):
        self.canvas = canvas
        self.field = field
        self.seeding = seeding_strategy
        self.length = length

    def generate_pattern(self, style=None, **kwargs) -> StreamlinePattern:
        seeds = self.seeding.generate()
        streamlines = []
        magnitudes = []

        # Vectorized evaluation
        vectors = self.field(seeds)

        for i, seed in enumerate(seeds):
            v = vectors[i]
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-6:
                continue

            # Create a short segment
            v_unit = v / v_norm
            end_pos = seed + v_unit * self.length

            # Only include if both points are inside the canvas
            if self.canvas.contains(end_pos):
                streamlines.append(np.array([seed, end_pos]))
                magnitudes.append(np.array([v_norm, v_norm]))

        return StreamlinePattern(streamlines=streamlines, magnitudes=magnitudes, canvas=self.canvas)
