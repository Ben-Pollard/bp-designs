"""Strategies for Space Colonization growth, attraction, and topology."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import shapely
from shapely.geometry import Polygon as ShapelyPolygon

if TYPE_CHECKING:
    from bp_designs.core.geometry import Polygon
    from bp_designs.patterns.network import BranchNetwork


class GrowthStrategy(ABC):
    """Controls how growth vectors are calculated and modified."""

    @abstractmethod
    def refine_vectors(self, vectors: np.ndarray, network: BranchNetwork) -> np.ndarray:
        """Modify raw attraction vectors.

        Args:
            vectors: (N, 2) array of raw growth vectors
            network: Current branch network

        Returns:
            (N, 2) array of refined growth vectors
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

    def refine_vectors(self, vectors: np.ndarray, network: BranchNetwork) -> np.ndarray:
        norms = np.sqrt(np.sum(vectors**2, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.where(norms[:, None] > 0, vectors / norms[:, None], 0.0)
        return normalized * self.segment_length


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

    def refine_vectors(self, vectors: np.ndarray, network: BranchNetwork) -> np.ndarray:
        # Momentum implementation would require knowing which nodes are growing
        # to look up their previous segments. For now, we'll implement GridSnappedGrowth
        # as the first "Wild" demo.
        return vectors


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

    def refine_vectors(self, vectors: np.ndarray, network: BranchNetwork) -> np.ndarray:
        if vectors.size == 0:
            return vectors

        # Calculate angles of raw vectors
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        # Snap to nearest step
        snapped_angles = np.round(angles / self.angle_step) * self.angle_step

        # Reconstruct vectors
        snapped_vectors = np.column_stack([np.cos(snapped_angles), np.sin(snapped_angles)])

        return snapped_vectors * self.segment_length
