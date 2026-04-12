from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from bp_designs.core.field import Field
from bp_designs.core.geometry import Polygon


class IntegrationStrategy(ABC):
    """Abstract base class for numerical integration strategies."""

    @abstractmethod
    def step(self, field: Field, positions: np.ndarray, dt: float) -> np.ndarray:
        """Calculate next positions for N particles.

        Args:
            field: The vector field to integrate.
            positions: (N, 2) array of current positions.
            dt: Time step.

        Returns:
            (N, 2) array of next positions.
        """
        pass


class EulerIntegrator(IntegrationStrategy):
    """Simple Euler integration: pos + field(pos) * dt."""

    def step(self, field: Field, positions: np.ndarray, dt: float) -> np.ndarray:
        return positions + field(positions) * dt


class RK4Integrator(IntegrationStrategy):
    """4th-order Runge-Kutta integration for better stability."""

    def step(self, field: Field, positions: np.ndarray, dt: float) -> np.ndarray:
        k1 = field(positions)
        k2 = field(positions + 0.5 * dt * k1)
        k3 = field(positions + 0.5 * dt * k2)
        k4 = field(positions + dt * k3)

        return positions + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class SeedingStrategy(ABC):
    """Abstract base class for seeding strategies."""

    @abstractmethod
    def generate(self) -> np.ndarray:
        """Generate (N, 2) seed positions."""
        pass


class RandomSeeding(SeedingStrategy):
    """Uniform random seeding within bounds."""

    def __init__(self, bounds: np.ndarray, num_seeds: int = 100, seed: int = 0):
        self.bounds = bounds
        self.num_seeds = num_seeds
        self.rng = np.random.default_rng(seed)

    def generate(self) -> np.ndarray:
        return self.rng.uniform(self.bounds[0], self.bounds[1], (self.num_seeds, 2))


class GridSeeding(SeedingStrategy):
    """Regular grid seeding within bounds."""

    def __init__(self, bounds: np.ndarray, resolution: tuple[int, int] = (10, 10)):
        self.bounds = bounds
        self.resolution = resolution

    def generate(self) -> np.ndarray:
        x = np.linspace(self.bounds[0, 0], self.bounds[1, 0], self.resolution[0])
        y = np.linspace(self.bounds[0, 1], self.bounds[1, 1], self.resolution[1])
        xv, yv = np.meshgrid(x, y)
        return np.stack([xv.ravel(), yv.ravel()], axis=1)


class PoissonDiscSeeding(SeedingStrategy):
    """Blue-noise seeding using Poisson Disc sampling."""

    def __init__(self, bounds: np.ndarray, min_dist: float = 1.0, seed: int = 0, k: int = 30):
        self.bounds = bounds
        self.min_dist = min_dist
        self.rng = np.random.default_rng(seed)
        self.k = k

    def generate(self) -> np.ndarray:
        # Simple Bridson's algorithm implementation
        width = self.bounds[1, 0] - self.bounds[0, 0]
        height = self.bounds[1, 1] - self.bounds[0, 1]
        cell_size = self.min_dist / np.sqrt(2)

        cols = int(np.ceil(width / cell_size))
        rows = int(np.ceil(height / cell_size))

        grid = [None] * (cols * rows)

        def get_idx(p):
            c = int((p[0] - self.bounds[0, 0]) / cell_size)
            r = int((p[1] - self.bounds[0, 1]) / cell_size)
            return c, r

        active = []
        # Initial point
        p0 = self.rng.uniform(self.bounds[0], self.bounds[1])
        active.append(p0)
        c, r = get_idx(p0)
        grid[c + r * cols] = p0

        points = [p0]

        while active:
            idx = self.rng.integers(len(active))
            p = active[idx]
            found = False

            for _ in range(self.k):
                angle = self.rng.uniform(0, 2 * np.pi)
                radius = self.rng.uniform(self.min_dist, 2 * self.min_dist)
                pn = p + np.array([np.cos(angle), np.sin(angle)]) * radius

                if not (
                    self.bounds[0, 0] <= pn[0] <= self.bounds[1, 0]
                    and self.bounds[0, 1] <= pn[1] <= self.bounds[1, 1]
                ):
                    continue

                cn, rn = get_idx(pn)

                # Check neighbors
                too_close = False
                for i in range(max(0, cn - 2), min(cols, cn + 3)):
                    for j in range(max(0, rn - 2), min(rows, rn + 3)):
                        neighbor = grid[i + j * cols]
                        if neighbor is not None:
                            if np.linalg.norm(pn - neighbor) < self.min_dist:
                                too_close = True
                                break
                    if too_close:
                        break

                if not too_close:
                    points.append(pn)
                    active.append(pn)
                    grid[cn + rn * cols] = pn
                    found = True
                    break

            if not found:
                active.pop(idx)

        return np.array(points)


class TerminationStrategy(ABC):
    """Abstract base class for termination strategies."""

    @abstractmethod
    def should_terminate(
        self, positions: np.ndarray, step_count: int, ids: np.ndarray | None = None
    ) -> np.ndarray:
        """Check which positions should stop.

        Args:
            positions: (N, 2) array of current positions.
            step_count: Number of steps taken so far.
            ids: (N,) array of streamline IDs for the positions.

        Returns:
            (N,) boolean array where True means terminate.
        """
        pass

    def update(
        self, positions: np.ndarray, ids: np.ndarray | None = None, indices: np.ndarray | None = None
    ) -> None:
        """Optional: Update internal state with new positions.

        Args:
            positions: (N, 2) array of new positions to incorporate.
            ids: (N,) array of streamline IDs for the positions.
            indices: (N,) array of point indices within the streamlines.
        """
        pass


class FixedLengthTermination(TerminationStrategy):
    """Stop after a fixed number of steps."""

    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps

    def should_terminate(
        self, positions: np.ndarray, step_count: int, ids: np.ndarray | None = None
    ) -> np.ndarray:
        terminate = step_count >= self.max_steps
        return np.full(len(positions), terminate, dtype=bool)


class BoundaryTermination(TerminationStrategy):
    """Stop when leaving a boundary."""

    def __init__(self, boundary: Polygon):
        from shapely.geometry import Polygon as ShapelyPolygon

        self.boundary = boundary
        self.shapely_poly = ShapelyPolygon(boundary.coords)

    def should_terminate(
        self, positions: np.ndarray, step_count: int, ids: np.ndarray | None = None
    ) -> np.ndarray:
        from shapely import vectorized

        # vectorized.contains returns a boolean array for (N, 2) positions
        # We want to terminate if NOT contained
        is_inside = vectorized.contains(self.shapely_poly, positions[:, 0], positions[:, 1])
        return ~is_inside


class ProximityTermination(TerminationStrategy):
    """Stop when too close to existing points."""

    def __init__(
        self,
        existing_points: np.ndarray,
        min_dist: float = 1.0,
        existing_ids: np.ndarray | None = None,
        existing_indices: np.ndarray | None = None,
    ):
        from scipy.spatial import cKDTree

        self.points = existing_points
        self.ids = existing_ids if existing_ids is not None else np.full(len(existing_points), -1, dtype=int)
        self.indices = (
            existing_indices if existing_indices is not None else np.full(len(existing_points), -1, dtype=int)
        )
        self.tree = cKDTree(self.points) if len(self.points) > 0 else None
        self.min_dist = min_dist

    def get_collision_metadata(
        self, positions: np.ndarray, ids: np.ndarray | None = None
    ) -> list[dict | None]:
        """Get metadata about collisions for the given positions.

        Returns:
            List of dicts (one per position) containing 'id' and 'index' of the hit point,
            or None if no collision within min_dist.
        """
        if self.tree is None:
            return [None] * len(positions)

        k = min(10, len(self.points))
        # Ensure positions is at least 2D for cKDTree.query to return consistent shapes
        query_pos = positions
        if query_pos.ndim == 1:
            query_pos = query_pos[np.newaxis, :]

        dists, indices = self.tree.query(query_pos, k=k)

        # Ensure dists and indices are always (N, k)
        dists = np.asarray(dists).reshape(query_pos.shape[0], k)
        indices = np.asarray(indices).reshape(query_pos.shape[0], k)

        results = []
        for i in range(query_pos.shape[0]):
            my_id = ids[i] if ids is not None else -2
            found = None
            for d, idx in zip(dists[i], indices[i]):
                if d >= self.min_dist:
                    break
                neighbor_id = self.ids[idx]
                if neighbor_id != my_id:
                    found = {
                        "id": int(neighbor_id),
                        "index": int(self.indices[idx]),
                        "dist": float(d),
                    }
                    break
            results.append(found)

        return results

    def get_steering_metadata(
        self, positions: np.ndarray, radius: float, ids: np.ndarray | None = None
    ) -> list[dict | None]:
        """Get metadata about nearby points for steering.

        Returns:
            List of dicts (one per position) containing 'id', 'index', and 'dist' of the nearest point,
            or None if no point within radius.
        """
        if self.tree is None:
            return [None] * len(positions)

        k = min(10, len(self.points))
        query_pos = positions
        if query_pos.ndim == 1:
            query_pos = query_pos[np.newaxis, :]

        dists, indices = self.tree.query(query_pos, k=k)

        # Ensure dists and indices are always (N, k)
        dists = np.asarray(dists).reshape(query_pos.shape[0], k)
        indices = np.asarray(indices).reshape(query_pos.shape[0], k)

        results = []
        for i in range(query_pos.shape[0]):
            my_id = ids[i] if ids is not None else -2
            found = None
            for d, idx in zip(dists[i], indices[i]):
                if d >= radius:
                    break
                neighbor_id = self.ids[idx]
                if neighbor_id != my_id:
                    found = {
                        "id": int(neighbor_id),
                        "index": int(self.indices[idx]),
                        "dist": float(d),
                    }
                    break
            results.append(found)

        return results

    def should_terminate(
        self, positions: np.ndarray, step_count: int, ids: np.ndarray | None = None
    ) -> np.ndarray:
        # If no points yet, nothing to collide with
        if self.tree is None:
            return np.zeros(len(positions), dtype=bool)

        # Query for multiple neighbors to find potential collisions with OTHER streamlines
        # We query for k=10 to be safe, but we'll filter by ID.
        k = min(10, len(self.points))
        query_pos = positions
        if query_pos.ndim == 1:
            query_pos = query_pos[np.newaxis, :]

        dists, indices = self.tree.query(query_pos, k=k)

        # Ensure dists and indices are always (N, k)
        dists = np.asarray(dists).reshape(query_pos.shape[0], k)
        indices = np.asarray(indices).reshape(query_pos.shape[0], k)

        terminate = np.zeros(len(positions), dtype=bool)

        if ids is None:
            # Fallback to simple proximity if no IDs provided
            return dists[:, 0] < self.min_dist

        for i in range(query_pos.shape[0]):
            my_id = ids[i]
            # Check neighbors for this position
            for d, idx in zip(dists[i], indices[i]):
                if d >= self.min_dist:
                    break  # Neighbors are sorted by distance

                neighbor_id = self.ids[idx]
                if neighbor_id != my_id:
                    terminate[i] = True
                    break

        return terminate

    def update(
        self,
        positions: np.ndarray,
        ids: np.ndarray | None = None,
        indices: np.ndarray | None = None,
    ) -> None:
        """Add new points to the spatial index."""
        from scipy.spatial import cKDTree

        if len(positions) == 0:
            return

        self.points = np.concatenate([self.points, positions], axis=0)

        new_ids = ids if ids is not None else np.full(len(positions), -1, dtype=int)
        self.ids = np.concatenate([self.ids, new_ids], axis=0)

        new_indices = indices if indices is not None else np.full(len(positions), -1, dtype=int)
        self.indices = np.concatenate([self.indices, new_indices], axis=0)

        self.tree = cKDTree(self.points)

    def update_metadata(
        self,
        streamline_id: int,
        new_id: int,
        index_offset: int,
        reverse: bool,
        total_len: int,
    ) -> None:
        """Update metadata for points belonging to a streamline that was merged."""
        mask = self.ids == streamline_id
        self.ids[mask] = new_id
        if reverse:
            self.indices[mask] = (total_len - 1 - self.indices[mask]) + index_offset
        else:
            self.indices[mask] += index_offset


class CompositeTermination(TerminationStrategy):
    """Combines multiple termination strategies."""

    def __init__(self, strategies: list[TerminationStrategy]):
        self.strategies = strategies

    @property
    def min_dist(self) -> float:
        """Return the minimum distance of the first strategy that has it."""
        for s in self.strategies:
            if hasattr(s, "min_dist"):
                return s.min_dist
        return 0.0

    def should_terminate(
        self, positions: np.ndarray, step_count: int, ids: np.ndarray | None = None
    ) -> np.ndarray:
        terminate = np.zeros(len(positions), dtype=bool)
        for s in self.strategies:
            terminate |= s.should_terminate(positions, step_count, ids=ids)
        return terminate

    def update(
        self,
        positions: np.ndarray,
        ids: np.ndarray | None = None,
        indices: np.ndarray | None = None,
    ) -> None:
        for s in self.strategies:
            s.update(positions, ids=ids, indices=indices)

    def get_collision_metadata(
        self, positions: np.ndarray, ids: np.ndarray | None = None
    ) -> list[dict | None]:
        for s in self.strategies:
            if hasattr(s, "get_collision_metadata"):
                return s.get_collision_metadata(positions, ids=ids)
        return [None] * len(positions)

    def get_steering_metadata(
        self, positions: np.ndarray, radius: float, ids: np.ndarray | None = None
    ) -> list[dict | None]:
        for s in self.strategies:
            if hasattr(s, "get_steering_metadata"):
                return s.get_steering_metadata(positions, radius, ids=ids)
        return [None] * len(positions)

    def update_metadata(
        self,
        streamline_id: int,
        new_id: int,
        index_offset: int,
        reverse: bool,
        total_len: int,
    ) -> None:
        for s in self.strategies:
            if hasattr(s, "update_metadata"):
                s.update_metadata(streamline_id, new_id, index_offset, reverse, total_len)


class JoinStrategy(ABC):
    """Abstract base class for streamline joining strategies."""

    @abstractmethod
    def should_join(
        self,
        current_pos: np.ndarray,
        current_dir: np.ndarray,
        collision_metadata: dict,
        target_streamline: np.ndarray,
    ) -> bool:
        """Decide if a collision should result in a join.

        Args:
            current_pos: (2,) array of current position.
            current_dir: (2,) normalized direction vector.
            collision_metadata: Dict from ProximityTermination.get_collision_metadata.
            target_streamline: (M, 2) array of points in the target streamline.

        Returns:
            True if join should occur.
        """
        pass


class AngleJoinStrategy(JoinStrategy):
    """Join based on angular alignment and endpoint proximity."""

    def __init__(self, max_angle_deg: float = 30.0, endpoint_only: bool = True):
        self.max_angle_cos = np.cos(np.radians(max_angle_deg))
        self.endpoint_only = endpoint_only

    def should_join(
        self,
        current_pos: np.ndarray,
        current_dir: np.ndarray,
        collision_metadata: dict,
        target_streamline: np.ndarray,
    ) -> bool:
        idx = collision_metadata["index"]
        n = len(target_streamline)

        if n < 2:
            return False

        is_endpoint = idx == 0 or idx == n - 1
        if self.endpoint_only and not is_endpoint:
            return False

        # Calculate target tangent at hit point
        if idx == 0:
            target_dir = target_streamline[1] - target_streamline[0]
        elif idx == n - 1:
            target_dir = target_streamline[n - 1] - target_streamline[n - 2]
        else:
            # Average tangent for internal points (if allowed)
            target_dir = target_streamline[idx + 1] - target_streamline[idx - 1]

        target_dir_norm = np.linalg.norm(target_dir)
        if target_dir_norm < 1e-6:
            return False
        target_dir /= target_dir_norm

        # Check alignment (dot product)
        # We check both directions because we might be joining head-to-head or tail-to-head
        alignment = np.abs(np.dot(current_dir, target_dir))
        if alignment <= self.max_angle_cos:
            return False

        # Check join segment alignment to prevent sharp kinks
        target_pt = target_streamline[idx]
        join_vec = target_pt - current_pos
        join_dist = np.linalg.norm(join_vec)

        if join_dist > 1e-6:
            join_dir = join_vec / join_dist
            # The join segment should be reasonably aligned with the current direction.
            # However, if the join distance is very small relative to a typical step,
            # we allow it even if the angle is sharp, as it won't be a visible kink.
            join_alignment = np.dot(current_dir, join_dir)

            # If the join is "backwards" or very sideways, and not extremely close, reject it.
            # We use a threshold that allows for some lateral movement to meet the line.
            # We are very forgiving if the distance is small (e.g. < 1.0 units)
            if join_alignment < 0.0 and join_dist > 1.0:
                return False

        return True
