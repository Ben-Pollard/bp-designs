from __future__ import annotations

import numpy as np

from bp_designs.core.field import Field
from bp_designs.core.generator import Generator
from bp_designs.core.geometry import Canvas
from bp_designs.generators.flow.strategies import IntegrationStrategy, SeedingStrategy, TerminationStrategy
from bp_designs.patterns.flow import StreamlinePattern


class FlowGenerator(Generator):
    """Orchestrates streamline generation using fields and strategies."""

    def __init__(
        self,
        canvas: Canvas,
        field: Field,
        seeding_strategy: SeedingStrategy,
        integration_strategy: IntegrationStrategy,
        termination_strategy: TerminationStrategy,
        dt: float = 0.1
    ):
        self.canvas = canvas
        self.field = field
        self.seeding_strategy = seeding_strategy
        self.integration_strategy = integration_strategy
        self.termination_strategy = termination_strategy
        self.dt = dt

    def generate_pattern(self, **kwargs) -> StreamlinePattern:
        """Generate streamlines by integrating the field from seed points."""
        # Store kwargs in the pattern as per Generator interface
        pattern = self._generate_streamlines()
        pattern.render_params.update(kwargs)
        pattern.canvas = self.canvas
        return pattern

    def _generate_streamlines(self) -> StreamlinePattern:
        seeds = self.seeding_strategy.generate()
        num_seeds = len(seeds)

        # active_streamlines stores the points for each streamline
        # We'll use a list of lists for now, but we'll integrate in a vectorized way
        streamlines = [[p] for p in seeds]
        active_indices = list(range(num_seeds))
        current_positions = seeds.copy()

        step_count = 0

        while active_indices:
            step_count += 1

            # Vectorized integration step for all active streamlines
            next_positions = self.integration_strategy.step(
                self.field, current_positions[active_indices], self.dt
            )

            # Update positions and check termination
            still_active = []
            for i, idx in enumerate(active_indices):
                new_pos = next_positions[i]
                streamlines[idx].append(new_pos)

                # Check termination for this specific streamline
                # Note: TerminationStrategy.should_terminate takes (N, 2)
                # We pass the current point as a (1, 2) array
                if not self.termination_strategy.should_terminate(new_pos[None, :], step_count):
                    still_active.append(idx)
                    current_positions[idx] = new_pos

            active_indices = still_active

        # Convert lists of points to numpy arrays
        final_streamlines = [np.array(s) for s in streamlines]
        return StreamlinePattern(streamlines=final_streamlines)
