from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from bp_designs.core.field import Field
from bp_designs.core.generator import Generator
from bp_designs.core.geometry import Canvas
from bp_designs.generators.flow.strategies import IntegrationStrategy, SeedingStrategy, TerminationStrategy
from bp_designs.patterns.flow import FlowStyle, StreamlinePattern


class FlowConfig(BaseModel):
    """Structural parameters for flow field generation."""
    dt: float = 0.1
    max_steps: int = 100
    min_dist: float = 0.0  # 0.0 means no proximity termination
    seed_dist: float = 10.0

class FlowGenerator(Generator):
    """Orchestrates streamline generation using fields and strategies."""

    def __init__(
        self,
        canvas: Canvas,
        field: Field,
        seeding_strategy: SeedingStrategy,
        integration_strategy: IntegrationStrategy,
        termination_strategy: TerminationStrategy,
        config: FlowConfig | None = None,
        dt: float | None = None # Deprecated: use config
    ):
        self.canvas = canvas
        self.field = field
        self.seeding_strategy = seeding_strategy
        self.integration_strategy = integration_strategy
        self.termination_strategy = termination_strategy

        if config is None:
            config = FlowConfig(dt=dt if dt is not None else 0.1)
        self.config = config
        self.dt = self.config.dt

    def generate_pattern(self, style: FlowStyle | dict | None = None, **kwargs) -> StreamlinePattern:
        """Generate streamlines by integrating the field from seed points.

        Args:
            style: A FlowStyle object or a dictionary of style parameters.
            **kwargs: Additional style parameters (merged with style).
        """
        # Merge style and kwargs into a single dictionary for the pattern
        render_params = {}
        if isinstance(style, FlowStyle):
            render_params = style.model_dump()
        elif isinstance(style, dict):
            render_params = style.copy()

        render_params.update(kwargs)

        pattern = self._generate_streamlines()
        pattern.render_params = render_params
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

        # Store magnitudes for each streamline
        magnitudes = [[0.0] for _ in seeds]  # Initial magnitude is 0 or field at seed

        while active_indices:
            step_count += 1

            # Vectorized integration step for all active streamlines
            active_pos = current_positions[active_indices]

            # Get field values to compute magnitudes
            field_vals = self.field(active_pos)
            mags = np.linalg.norm(field_vals, axis=1)

            next_positions = self.integration_strategy.step(self.field, active_pos, self.dt)

            # Check termination for all active streamlines at once
            # Pass the original streamline indices as IDs
            active_ids = np.array(active_indices)
            should_stop = self.termination_strategy.should_terminate(next_positions, step_count, ids=active_ids)

            # Update positions and filter active indices
            still_active = []
            for i, (idx, stop) in enumerate(zip(active_indices, should_stop)):
                new_pos = next_positions[i]
                streamlines[idx].append(new_pos)
                magnitudes[idx].append(mags[i])

                if not stop:
                    still_active.append(idx)
                    current_positions[idx] = new_pos

            # Update termination strategy with new points if needed (e.g. for proximity)
            # Only add points for streamlines that are NOT stopping this step
            self.termination_strategy.update(next_positions[~should_stop], ids=active_ids[~should_stop])

            active_indices = still_active

        # Convert lists of points to numpy arrays
        final_streamlines = [np.array(s) for s in streamlines]
        final_magnitudes = [np.array(m) for m in magnitudes]
        return StreamlinePattern(streamlines=final_streamlines, magnitudes=final_magnitudes)
