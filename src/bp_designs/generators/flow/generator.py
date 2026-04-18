from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from bp_designs.core.field import Field
from bp_designs.core.generator import Generator
from bp_designs.core.geometry import Canvas
from bp_designs.generators.flow.strategies import (
    IntegrationStrategy,
    JoinStrategy,
    ProximityTermination,
    SeedingStrategy,
    TerminationStrategy,
)
from bp_designs.patterns.flow import StreamlinePattern

if TYPE_CHECKING:
    from bp_designs.patterns.flow import FlowStyle


class FlowConfig(BaseModel):
    """Structural parameters for flow field generation."""

    model_config = {"arbitrary_types_allowed": True}

    dt: float = 0.1
    max_steps: int = 100
    min_dist: float = 0.0  # 0.0 means no proximity termination
    seed_dist: float = 10.0
    join_strategy: JoinStrategy | None = None
    steering_radius: float = 0.0  # 0.0 means no steering
    steering_strength: float = 0.5
    steering_lookahead: float = 0.0  # Units to look ahead on target streamline


@dataclass
class _Streamline:
    """Internal state for a single streamline during generation."""

    id: int
    points: list[np.ndarray]
    magnitudes: list[float]
    is_active: bool = True

    @property
    def last_pos(self) -> np.ndarray:
        return self.points[-1]

    def add_point(self, pos: np.ndarray, mag: float):
        self.points.append(pos)
        self.magnitudes.append(mag)


class FlowGenerator(Generator):
    """Orchestrates streamline generation using fields and strategies.

    Refactored for modularity and maintainability, separating integration,
    steering, and collision logic.
    """

    def __init__(
        self,
        canvas: Canvas,
        field: Field,
        seeding_strategy: SeedingStrategy,
        integration_strategy: IntegrationStrategy,
        termination_strategy: TerminationStrategy,
        join_strategy: JoinStrategy | None = None,
        config: FlowConfig | None = None,
        dt: float | None = None,  # Deprecated: use config
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

        # Override config join strategy if provided in constructor
        if join_strategy is not None:
            self.config.join_strategy = join_strategy

    def generate_pattern(self, style: FlowStyle | dict | None = None, **kwargs) -> StreamlinePattern:
        """Generate streamlines by integrating the field from seed points."""
        # 1. Initialize state
        streamlines = self._initialize_generation()

        # 2. Iterative growth
        for step in range(self.config.max_steps):
            active = [s for s in streamlines.values() if s.is_active]
            if not active:
                break

            self._execute_step(active, step, streamlines)

        # 3. Assemble pattern
        final_streamlines = [s.points for s in streamlines.values()]
        final_magnitudes = [s.magnitudes for s in streamlines.values()]

        # Pass style to the pattern via render_params
        from bp_designs.patterns.flow import FlowStyle

        render_params = {}
        if isinstance(style, FlowStyle):
            render_params = style.model_dump()
        elif isinstance(style, dict):
            render_params = style

        return StreamlinePattern(
            streamlines=final_streamlines,
            magnitudes=final_magnitudes,
            canvas=self.canvas,
            render_params=render_params,
        )

        for i, meta in enumerate(steering_metadata):
            if meta is None:
                continue

            target_id = meta["id"]
            target_idx = meta["index"]
            dist = meta["dist"]

            target_s = all_streamlines.get(target_id)
            if not target_s or len(target_s.points) < 2:
                continue

            # 1. Early rejection: if the neighbor is flowing in a completely
            # different direction, don't steer towards it.
            # We look at the tangent at the nearest point.
            nearest_tangent = self._calculate_tangent(target_s.points, target_idx)
            if nearest_tangent is None:
                continue

            # Align tangent with current flow
            if np.dot(field_vectors[i], nearest_tangent) < 0:
                nearest_tangent = -nearest_tangent

            # Check alignment gate (default 60 deg if no join_strategy)
            max_angle_cos = 0.5  # cos(60 deg)
            from bp_designs.generators.flow.strategies import AngleJoinStrategy

            if isinstance(self.config.join_strategy, AngleJoinStrategy):
                max_angle_cos = self.config.join_strategy.max_angle_cos

            field_mag = np.linalg.norm(field_vectors[i])
            if field_mag < 1e-6:
                continue
            field_dir = field_vectors[i] / field_mag

            if np.dot(field_dir, nearest_tangent) < max_angle_cos:
                # Flow is too incompatible, skip steering to avoid kinks
                continue

            # 2. Look-ahead (Pursuit Steering)
            lookahead_idx = target_idx
            target_pt = target_s.points[target_idx]
            target_tangent = nearest_tangent

            if self.config.steering_lookahead > 0:
                # Traverse forward along target streamline to find lookahead point
                accum_dist = 0.0
                j = target_idx
                while j < len(target_s.points) - 1:
                    seg_vec = target_s.points[j + 1] - target_s.points[j]
                    seg_len = np.linalg.norm(seg_vec)
                    if accum_dist + seg_len > self.config.steering_lookahead:
                        # Interpolate to find exact lookahead within existing points
                        rem = self.config.steering_lookahead - accum_dist
                        target_pt = target_s.points[j] + seg_vec * (rem / seg_len)
                        target_tangent = self._calculate_tangent(target_s.points, j)
                        if target_tangent is not None and np.dot(field_dir, target_tangent) < 0:
                            target_tangent = -target_tangent
                        accum_dist = self.config.steering_lookahead  # Mark as done
                        break
                    accum_dist += seg_len
                    j += 1

                # If we reached the end of the streamline but still haven't
                # reached the lookahead distance, extrapolate along the last tangent
                if accum_dist < self.config.steering_lookahead:
                    last_tangent = self._calculate_tangent(target_s.points, len(target_s.points) - 1)
                    if last_tangent is not None:
                        if np.dot(field_dir, last_tangent) < 0:
                            last_tangent = -last_tangent
                        rem = self.config.steering_lookahead - accum_dist
                        target_pt = target_s.points[-1] + last_tangent * rem
                        target_tangent = last_tangent

            if target_tangent is None:
                target_tangent = nearest_tangent

            # 3. Blending and Pulling
            # Blending factor (alpha=1 at min_dist, alpha=0 at steering_radius)
            min_d = self.termination_strategy.min_dist
            r = self.config.steering_radius

            # Smoothstep (cubic) blending for gentler approach
            t = 1.0 - (dist - min_d) / (r - min_d)
            t = np.clip(t, 0.0, 1.0)
            alpha = (3 * t**2 - 2 * t**3) * self.config.steering_strength

            # Blend field direction with target tangent
            blended_dir = (1.0 - alpha) * field_dir + alpha * target_tangent

            # Weighted pull towards lookahead point to close lateral gap
            pull_vec = target_pt - positions[i]
            p_dist = np.linalg.norm(pull_vec)
            if p_dist > 1e-6:
                alignment = np.clip(np.dot(field_dir, target_tangent), 0.0, 1.0)
                # Damped pull weight to avoid oscillations
                # We limit the lateral pull to prevent sharp kinks
                pull_weight = alpha * (0.2 + 0.3 * alignment) * np.clip(p_dist / r, 0.0, 0.5)
                blended_dir += (pull_vec / p_dist) * pull_weight

            b_norm = np.linalg.norm(blended_dir)
            if b_norm > 1e-6:
                new_dir = blended_dir / b_norm

                # Limit the maximum angle change per step to ensure smoothness
                # Max 15 degrees per unit distance
                max_angle = np.radians(15.0) * self.dt
                cos_angle = np.clip(np.dot(field_dir, new_dir), -1.0, 1.0)
                if cos_angle < np.cos(max_angle):
                    # Need to rotate field_dir towards new_dir by max_angle
                    # 2D rotation formula
                    angle = np.arccos(cos_angle)
                    scale = max_angle / angle
                    # Slerp-like rotation
                    new_dir = field_dir * np.cos(max_angle) + (new_dir - field_dir * cos_angle) * (
                        np.sin(max_angle) / np.sin(angle)
                    )

                steered_vectors[i] = new_dir * field_mag

        return positions + steered_vectors * self.dt

    def _handle_collisions(
        self,
        active: list[_Streamline],
        next_positions: np.ndarray,
        terminate_mask: np.ndarray,
        all_streamlines: dict[int, _Streamline],
    ):
        """Process terminations and joins for the current step."""
        collision_metadata = [None] * len(active)
        if self.config.join_strategy and hasattr(self.termination_strategy, "get_collision_metadata"):
            collision_metadata = self.termination_strategy.get_collision_metadata(
                next_positions, np.array([s.id for s in active])
            )

        for i, s in enumerate(active):
            if not terminate_mask[i]:
                # Normal step: add point and continue
                s.add_point(next_positions[i], self._get_magnitude(next_positions[i]))
                continue

            # Termination triggered - check if it's a join
            meta = collision_metadata[i]
            if meta and self.config.join_strategy:
                target_s = all_streamlines.get(meta["id"])
                if target_s:
                    current_dir = next_positions[i] - s.last_pos
                    d_norm = np.linalg.norm(current_dir)
                    if d_norm > 1e-6:
                        current_dir /= d_norm
                        if self.config.join_strategy.should_join(
                            next_positions[i], current_dir, meta, np.array(target_s.points)
                        ):
                            # JOIN: Merge source into target and remove source
                            self._merge_streamlines(s, target_s, meta["index"], next_positions[i])
                            all_streamlines.pop(s.id)
                            s.is_active = False
                            continue

            # Regular termination (boundary or proximity without join)
            # Add the final point that triggered termination for continuity
            s.add_point(next_positions[i], self._get_magnitude(next_positions[i]))
            s.is_active = False

    def _merge_streamlines(
        self,
        source: _Streamline,
        target: _Streamline,
        target_index: int,
        collision_point: np.ndarray,
    ):
        """Merge source streamline into target streamline with Hermite bridge for G1 continuity."""
        # 1. Get endpoints and tangents
        p_source = source.points[-1]
        t_source = self._calculate_tangent(source.points, len(source.points) - 1)
        if t_source is None:
            t_source = np.array([1.0, 0.0])

        p_target = target.points[target_index]
        t_target = self._calculate_tangent(target.points, target_index)
        if t_target is None:
            t_target = np.array([1.0, 0.0])

        # Ensure tangents align with flow
        if np.dot(t_source, t_target) < 0:
            t_target = -t_target

        # 2. Decide if we need a bridge
        gap_vec = p_target - p_source
        gap_dist = np.linalg.norm(gap_vec)

        # If very close, just snap by dropping collision point
        if gap_dist < 0.2 * self.dt:
            mid_pts = []
        else:
            # Create smooth bridge using Hermite spline (cubic)
            # Length of tangents should be proportional to gap distance
            # for a natural curve.
            k = gap_dist * 0.5
            p0, v0 = p_source, t_source * k
            p1, v1 = p_target, t_target * k

            # Number of points in bridge depends on gap distance and dt
            n_pts = int(max(1, gap_dist / self.dt))
            if n_pts > 1:
                mid_pts = []
                for t in np.linspace(0, 1, n_pts + 2)[1:-1]:
                    # Cubic Hermite basis functions
                    h00 = 2 * t**3 - 3 * t**2 + 1
                    h10 = t**3 - 2 * t**2 + t
                    h01 = -2 * t**3 + 3 * t**2
                    h11 = t**3 - t**2

                    pt = h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1
                    mid_pts.append(pt)
            else:
                # Just use the collision point or target point
                mid_pts = [collision_point] if gap_dist > self.dt else []

        # 3. Assemble new streamline
        source_pts = source.points
        source_mags = source.magnitudes
        target_pts = target.points
        target_mags = target.magnitudes

        mid_mags = [self._get_magnitude(p) for p in mid_pts]

        if target_index == 0:
            # Join to start: [Source] -> [Bridge] -> [Target]
            # Handle potential duplicate with target start
            target_start_idx = 0
            if len(mid_pts) > 0:
                if np.linalg.norm(mid_pts[-1] - target_pts[0]) < 0.1 * self.dt:
                    target_start_idx = 1
            elif np.linalg.norm(source_pts[-1] - target_pts[0]) < 0.1 * self.dt:
                target_start_idx = 1

            new_pts = source_pts + mid_pts + target_pts[target_start_idx:]
            new_mags = source_mags + mid_mags + target_mags[target_start_idx:]

            if isinstance(self.termination_strategy, ProximityTermination):
                offset = len(source_pts) + len(mid_pts) - target_start_idx
                self.termination_strategy.update_metadata(
                    target.id, target.id, offset, False, len(target_pts)
                )
                self.termination_strategy.update_metadata(source.id, target.id, 0, False, len(source_pts))
                if len(mid_pts) > 0:
                    self.termination_strategy.update(
                        np.array(mid_pts),
                        np.full(len(mid_pts), target.id),
                        np.array([len(source_pts) + i for i in range(len(mid_pts))]),
                    )
        else:
            # Join to end: [Target] -> [Bridge] -> [Source Reversed]
            source_rev_pts = source_pts[::-1]
            source_rev_mags = source_mags[::-1]

            source_start_idx = 0
            if len(mid_pts) > 0:
                if np.linalg.norm(mid_pts[-1] - source_rev_pts[0]) < 0.1 * self.dt:
                    source_start_idx = 1
            elif np.linalg.norm(target_pts[-1] - source_rev_pts[0]) < 0.1 * self.dt:
                source_start_idx = 1

            new_pts = target_pts + mid_pts + source_rev_pts[source_start_idx:]
            new_mags = target_mags + mid_mags + source_rev_mags[source_start_idx:]

            if isinstance(self.termination_strategy, ProximityTermination):
                offset = len(target_pts) + len(mid_pts) - source_start_idx
                self.termination_strategy.update_metadata(source.id, target.id, offset, True, len(source_pts))
                if len(mid_pts) > 0:
                    self.termination_strategy.update(
                        np.array(mid_pts),
                        np.full(len(mid_pts), target.id),
                        np.array([len(target_pts) + i for i in range(len(mid_pts))]),
                    )

        target.points = new_pts
        target.magnitudes = new_mags

        target.points = new_pts
        target.magnitudes = new_mags

    def _finalize_pattern(self, streamlines: dict[int, _Streamline], **kwargs) -> StreamlinePattern:
        """Convert internal streamlines to the final Pattern object."""
        valid = [s for s in streamlines.values() if len(s.points) >= 2]
        return StreamlinePattern(
            streamlines=[np.array(s.points) for s in valid],
            magnitudes=[np.array(s.magnitudes) for s in valid],
            render_params=kwargs,
            canvas=self.canvas,
        )

    def _get_magnitude(self, pos: np.ndarray) -> float:
        """Helper to get field magnitude at a position."""
        return float(np.linalg.norm(self.field(pos[np.newaxis, :])))

    def _calculate_tangent(self, points: list[np.ndarray], index: int) -> np.ndarray | None:
        """Helper to calculate tangent at a specific point index."""
        n = len(points)
        if n < 2:
            return None

        if index == 0:
            tangent = points[1] - points[0]
        elif index == n - 1:
            tangent = points[n - 1] - points[n - 2]
        else:
            tangent = points[index + 1] - points[index - 1]

        t_norm = np.linalg.norm(tangent)
        return tangent / t_norm if t_norm > 1e-6 else None
