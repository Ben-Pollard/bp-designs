from __future__ import annotations

import numpy as np
from bp_designs.core.generator import Generator
from bp_designs.core.geometry import Canvas
from bp_designs.patterns.flow import StreamlinePattern
from bp_designs.generators.flow.strategies import (
    IntegrationStrategy,
    SeedingStrategy,
    RK4Integrator,
)
from .generator import FlowConfig


class ClassicFlowGenerator(Generator):
    """A standard (sequential) flow field generator for long streamlines.

    This follows the 'Classic' approach: seed a point, trace it as far as possible,
    optionally checking for collisions against previous lines. This is much simpler
    and more stable than the agent-based simulation for creating long, flowing textures.
    """

    def __init__(
        self,
        canvas: Canvas,
        field,
        seeding_strategy: SeedingStrategy,
        integration_strategy: IntegrationStrategy | None = None,
        config: FlowConfig | None = None,
        trace_both_ways: bool = True,
    ):
        self.canvas = canvas
        self.field = field
        self.seeding = seeding_strategy
        self.integration = integration_strategy or RK4Integrator()
        self.config = config or FlowConfig()
        self.trace_both_ways = trace_both_ways

    def generate_pattern(self, style=None, **kwargs) -> StreamlinePattern:
        seeds = self.seeding.generate()
        streamlines = []
        magnitudes = []

        # Spatial index for collision avoidance
        from scipy.spatial import cKDTree

        all_points_list = []
        tree = None

        for seed in seeds:
            # Check if seed is already too close to existing lines
            if tree is not None and self.config.min_dist > 0:
                dist, _ = tree.query(seed, k=1)
                if dist < self.config.min_dist:
                    continue

            # Trace forward
            fwd_path, fwd_mags = self._trace(seed, direction=1.0, tree=tree)

            if self.trace_both_ways:
                # Trace backward
                back_path, back_mags = self._trace(seed, direction=-1.0, tree=tree)

                # Combine (backward path is returned from seed outwards, so reverse it)
                full_path = np.vstack([back_path[::-1], fwd_path[1:]])
                full_mags = np.concatenate([back_mags[::-1], fwd_mags[1:]])
            else:
                full_path = fwd_path
                full_mags = fwd_mags

            if len(full_path) > 2:
                streamlines.append(full_path)
                magnitudes.append(full_mags)

                # Update spatial index with new points
                if self.config.min_dist > 0:
                    all_points_list.append(full_path)
                    all_points = np.concatenate(all_points_list, axis=0)
                    tree = cKDTree(all_points)

        return StreamlinePattern(streamlines=streamlines, magnitudes=magnitudes, canvas=self.canvas)

    def _trace(self, seed: np.ndarray, direction: float = 1.0, tree=None) -> tuple[np.ndarray, np.ndarray]:
        path = [seed]
        mags = [self._get_magnitude(seed)]
        current = seed

        effective_dt = self.config.dt * direction

        for _ in range(self.config.max_steps):
            # Step the integration
            next_pos = self.integration.step(self.field, current[np.newaxis, :], effective_dt)[0]

            # Bounds check
            if not self.canvas.contains(next_pos):
                break

            # Collision check
            if tree is not None and self.config.min_dist > 0:
                dist, _ = tree.query(next_pos, k=1)
                if dist < self.config.min_dist:
                    break

            path.append(next_pos)
            mags.append(self._get_magnitude(next_pos))
            current = next_pos

        return np.array(path), np.array(mags)

    def _get_magnitude(self, pos: np.ndarray) -> float:
        return float(np.linalg.norm(self.field(pos[np.newaxis, :])))
