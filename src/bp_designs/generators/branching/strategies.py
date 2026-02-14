"""Strategies for Space Colonization growth, attraction, and topology."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import shapely
from shapely.geometry import Polygon as ShapelyPolygon

if TYPE_CHECKING:
    from bp_designs.core.directions import DirectionVectors
    from bp_designs.core.geometry import Polygon
    from bp_designs.patterns.network import BranchNetwork


class GrowthStrategy(ABC):
    """Controls how growth vectors are calculated and modified."""

    @abstractmethod
    def refine_vectors(
        self,
        growth_vectors: DirectionVectors,
        network: BranchNetwork,
        growth_node_indices: np.ndarray,
    ) -> np.ndarray:
        """Modify raw attraction vectors.

        Args:
            growth_vectors: DirectionVectors object containing pre-calculated vectors, norms, and directions
            network: Current branch network
            growth_node_indices: (K,) indices of nodes in the network that are growing

        Returns:
            (K, 2) array of refined growth vectors
        """
        pass


class AttractionStrategy(ABC):
    """Controls how attraction points are initialized and updated."""

    @abstractmethod
    def initialize(self, num: int, boundary: Polygon, rng: np.random.Generator) -> np.ndarray:
        """Initialize attraction points.

        Args:
            num: Number of points to generate
            boundary: Polygon defining the region
            rng: Random number generator

        Returns:
            (M, 2) array of attraction point positions
        """
        pass

    @abstractmethod
    def update(self, attractions: np.ndarray, network: BranchNetwork, rng: np.random.Generator) -> np.ndarray:
        """Update attraction points per iteration.

        Args:
            attractions: Current (M, 2) array of attraction points
            network: Current branch network
            rng: Random number generator

        Returns:
            (K, 2) array of updated attraction points
        """
        pass


class TopologyStrategy(ABC):
    """Controls how new nodes are connected to the network."""

    @abstractmethod
    def extend(
        self,
        network: BranchNetwork,
        new_positions: np.ndarray,
        parents: np.ndarray,
        timestamp: int,
        velocities: np.ndarray | None = None,
    ) -> BranchNetwork:
        """Define how new nodes connect to the network.

        Args:
            network: Current branch network
            new_positions: (K, 2) array of new node positions
            parents: (K,) array of parent node indices
            timestamp: Current iteration timestamp

        Returns:
            Updated BranchNetwork
        """
        pass


class DefaultGrowth(GrowthStrategy):
    """Standard growth: normalized vectors multiplied by segment length."""

    def __init__(self, segment_length: float = 2.0):
        self.segment_length = segment_length

    def refine_vectors(
        self,
        growth_vectors: DirectionVectors,
        network: BranchNetwork,
        growth_node_indices: np.ndarray,
    ) -> np.ndarray:
        # Default behavior: just return the pre-calculated vectors for growing nodes
        return growth_vectors.vectors[growth_node_indices]


class DefaultAttraction(AttractionStrategy):
    """Standard attraction: static points, rejection sampling in boundary."""

    def initialize(self, num: int, boundary: Polygon, rng: np.random.Generator) -> np.ndarray:
        if num <= 0:
            return np.array([], dtype=float).reshape(0, 2)

        bounds = boundary.bounds()
        xmin, ymin, xmax, ymax = bounds
        shapely_poly = ShapelyPolygon(boundary.coords)

        points = []
        batch_size = min(num * 2, 1000)
        max_attempts = 100
        attempts = 0

        while len(points) < num and attempts < max_attempts:
            attempts += 1
            x = rng.uniform(xmin, xmax, batch_size)
            y = rng.uniform(ymin, ymax, batch_size)
            candidates = np.column_stack([x, y])
            candidate_points = shapely.points(candidates)
            mask = shapely.contains(shapely_poly, candidate_points)
            valid_points = candidates[mask]
            points.extend(valid_points)

        return np.array(points[:num])

    def update(self, attractions: np.ndarray, network: BranchNetwork, rng: np.random.Generator) -> np.ndarray:
        # Default behavior: attractions are static, they don't move or change
        # (Colonization/removal is handled by the core loop)
        return attractions


class DefaultTopology(TopologyStrategy):
    """Standard topology: strict parent-child tree."""

    def extend(
        self,
        network: BranchNetwork,
        new_positions: np.ndarray,
        parents: np.ndarray,
        timestamp: int,
        velocities: np.ndarray | None = None,
    ) -> BranchNetwork:
        from bp_designs.patterns.network import BranchNetwork

        num_new_nodes = new_positions.shape[0]
        if num_new_nodes == 0:
            return network

        next_node_id = network.node_ids.max() + 1
        new_node_ids = np.arange(next_node_id, next_node_id + num_new_nodes)
        new_timestamps = np.full(num_new_nodes, timestamp, dtype=np.int16)

        # Map parent indices to actual node IDs
        parent_ids = network.node_ids[parents]

        return BranchNetwork(
            node_ids=np.hstack([network.node_ids, new_node_ids]),
            positions=np.vstack([network.positions, new_positions]),
            parents=np.hstack([network.parents, parent_ids]),
            timestamps=np.hstack([network.timestamps, new_timestamps]),
            velocities=(
                np.vstack([network.velocities, velocities])
                if network.velocities is not None and velocities is not None
                else velocities
            ),
            canvas=network.canvas,
            pattern_bounds=network.pattern_bounds,
        )


class MomentumGrowth(GrowthStrategy):
    """Growth with momentum: blends current direction with previous segment direction."""

    def __init__(self, segment_length: float = 2.0, momentum: float = 0.5):
        """Initialize momentum growth.

        Args:
            segment_length: Length of each growth segment
            momentum: Weight of previous direction (0.0 to 1.0)
        """
        self.segment_length = segment_length
        self.momentum = momentum

    def refine_vectors(
        self,
        growth_vectors: DirectionVectors,
        network: BranchNetwork,
        growth_node_indices: np.ndarray,
    ) -> np.ndarray:
        if growth_node_indices.size == 0:
            return np.zeros((0, 2))

        # Use pre-calculated directions for growing nodes
        current_dirs = growth_vectors.directions[growth_node_indices]

        # Get previous directions for each growing node from the velocities array
        if network.velocities is not None:
            prev_dirs = network.velocities[growth_node_indices]
            # Normalize prev_dirs to ensure consistent blending
            p_norms = np.sqrt(np.sum(prev_dirs**2, axis=1))
            with np.errstate(divide="ignore", invalid="ignore"):
                prev_dirs = np.where(p_norms[:, None] > 0, prev_dirs / p_norms[:, None], 0.0)
        else:
            prev_dirs = current_dirs

        # Blend directions (Exponential Decay)
        blended = (1.0 - self.momentum) * current_dirs + self.momentum * prev_dirs

        # Re-normalize the blended direction
        b_norms = np.sqrt(np.sum(blended**2, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            refined_dirs = np.where(b_norms[:, None] > 0, blended / b_norms[:, None], 0.0)

        return refined_dirs * self.segment_length


class GridSnappedGrowth(GrowthStrategy):
    """Growth snapped to specific angles."""

    def __init__(self, segment_length: float = 2.0, angles: int = 8):
        """Initialize grid-snapped growth.

        Args:
            segment_length: Length of each growth segment
            angles: Number of allowed directions (e.g., 4 for 90deg, 8 for 45deg)
        """
        self.segment_length = segment_length
        self.angle_step = 2 * np.pi / angles

    def refine_vectors(
        self,
        growth_vectors: DirectionVectors,
        network: BranchNetwork,
        growth_node_indices: np.ndarray,
    ) -> np.ndarray:
        if growth_node_indices.size == 0:
            return np.zeros((0, 2))

        vectors = growth_vectors.vectors[growth_node_indices]

        # Calculate angles of raw vectors
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        # Snap to nearest step
        snapped_angles = np.round(angles / self.angle_step) * self.angle_step

        # Reconstruct vectors
        snapped_vectors = np.column_stack([np.cos(snapped_angles), np.sin(snapped_angles)])

        return snapped_vectors * self.segment_length


class ObstacleAvoidanceGrowth(GrowthStrategy):
    """Growth that steers around geometric obstacles."""

    def __init__(
        self,
        obstacles: list[Polygon],
        segment_length: float = 2.0,
        num_samples: int = 8,
        max_angle: float = np.pi / 2,
    ):
        """Initialize obstacle avoidance.

        Args:
            obstacles: List of Polygon objects to avoid
            segment_length: Length of each growth segment
            num_samples: Number of alternative directions to sample if blocked
            max_angle: Maximum angle to deviate from target direction
        """
        self.segment_length = segment_length
        self.num_samples = num_samples
        self.max_angle = max_angle
        # Pre-convert to shapely for performance
        self.shapely_obstacles = []
        for p in obstacles:
            if hasattr(p, "coords"):
                self.shapely_obstacles.append(ShapelyPolygon(p.coords))
            elif hasattr(p, "polylines"):
                # Use the first polyline as the boundary
                self.shapely_obstacles.append(ShapelyPolygon(p.polylines[0]))

    def refine_vectors(
        self,
        growth_vectors: DirectionVectors,
        network: BranchNetwork,
        growth_node_indices: np.ndarray,
    ) -> np.ndarray:
        if growth_node_indices.size == 0 or not self.shapely_obstacles:
            return growth_vectors.vectors[growth_node_indices]

        vectors = growth_vectors.vectors[growth_node_indices]
        refined = vectors.copy()
        node_positions = network.positions[growth_node_indices]

        for i in range(len(vectors)):
            start = node_positions[i]
            target_v = vectors[i]

            # Check if direct path is blocked
            if self._is_blocked(start, start + target_v):
                # Try alternative directions
                best_v = None

                # Sample angles to left and right
                angles = np.linspace(-self.max_angle, self.max_angle, self.num_samples)
                # Sort by absolute angle to prefer smaller deviations
                angles = angles[np.argsort(np.abs(angles))]

                base_angle = np.arctan2(target_v[1], target_v[0])

                for angle_offset in angles:
                    angle = base_angle + angle_offset
                    test_v = np.array([np.cos(angle), np.sin(angle)]) * self.segment_length
                    if not self._is_blocked(start, start + test_v):
                        best_v = test_v
                        break

                if best_v is not None:
                    refined[i] = best_v
                else:
                    # If all samples blocked, stop growth (zero vector)
                    refined[i] = np.zeros(2)

        return refined

    def _is_blocked(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Check if segment intersects any obstacle."""
        from shapely.geometry import LineString

        line = LineString([start, end])
        for obs in self.shapely_obstacles:
            if line.intersects(obs):
                return True
        return False


class DriftAttraction(AttractionStrategy):
    """Attraction points that drift in a constant direction."""

    def __init__(self, drift_vector: np.ndarray = np.array([1.0, 0.0])):
        self.drift_vector = drift_vector

    def initialize(self, num: int, boundary: Polygon, rng: np.random.Generator) -> np.ndarray:
        return DefaultAttraction().initialize(num, boundary, rng)

    def update(self, attractions: np.ndarray, network: BranchNetwork, rng: np.random.Generator) -> np.ndarray:
        if attractions.size == 0:
            return attractions
        return attractions + self.drift_vector


class VortexAttraction(AttractionStrategy):
    """Attraction points that rotate around a center."""

    def __init__(self, center: np.ndarray, strength: float = 0.05):
        self.center = center
        self.strength = strength

    def initialize(self, num: int, boundary: Polygon, rng: np.random.Generator) -> np.ndarray:
        return DefaultAttraction().initialize(num, boundary, rng)

    def update(self, attractions: np.ndarray, network: BranchNetwork, rng: np.random.Generator) -> np.ndarray:
        if attractions.size == 0:
            return attractions

        # Vector from center to points
        diff = attractions - self.center

        # Perpendicular vector (rotate 90 deg)
        perp = np.column_stack([-diff[:, 1], diff[:, 0]])

        # Normalize and apply strength
        norms = np.sqrt(np.sum(perp**2, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            perp_norm = np.where(norms[:, None] > 0, perp / norms[:, None], 0.0)

        return attractions + perp_norm * self.strength
