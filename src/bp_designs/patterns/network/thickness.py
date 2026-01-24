"""Thickness strategies for branching networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bp_designs.patterns.network.base import BranchNetwork


class BranchThicknessStrategy:
    """Base class for computing branch thickness from network structure."""

    def compute_thickness(self, network: BranchNetwork) -> np.ndarray:
        """Compute thickness value for each node in the network.

        Args:
            network: BranchNetwork with semantic information

        Returns:
            (N,) array of thickness values for each node
        """
        raise NotImplementedError

    @staticmethod
    def from_name(name: str, **kwargs) -> BranchThicknessStrategy:
        """Factory method to create strategy by name."""
        if name == "timestamp":
            return TimestampThickness(**kwargs)
        elif name == "hierarchy":
            return HierarchyThickness(**kwargs)
        elif name == "descendant":
            return DescendantThickness(**kwargs)
        else:
            raise ValueError(f"Unknown thickness strategy: {name}")


class TimestampThickness(BranchThicknessStrategy):
    """Thickness based on node age/timestamp - older branches are thicker."""

    def __init__(
        self,
        min_thickness: float = 0.5,
        max_thickness: float = 5.0,
        power: float = 1.0,
        **kwargs,
    ):
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.power = power

    def compute_thickness(self, network: BranchNetwork) -> np.ndarray:
        """Compute thickness based on inverse timestamp (older = thicker)."""
        node_timestamps = network.timestamps
        if len(node_timestamps) == 0:
            return np.array([])

        min_time = node_timestamps.min()
        max_time = node_timestamps.max()

        if max_time > min_time:
            # Normalize to 0..1 where 1 is oldest (min_time)
            age_normalized = (max_time - node_timestamps) / (max_time - min_time)
        else:
            age_normalized = np.ones_like(node_timestamps, dtype=float)

        thickness = self.min_thickness + (age_normalized**self.power) * (
            self.max_thickness - self.min_thickness
        )
        return thickness


class HierarchyThickness(BranchThicknessStrategy):
    """Thickness based on distance from root - deeper branches are thinner."""

    def __init__(
        self,
        min_thickness: float = 0.5,
        max_thickness: float = 5.0,
        power: float = 1.0,
        **kwargs,
    ):
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.power = power

    def compute_thickness(self, network: BranchNetwork) -> np.ndarray:
        """Compute thickness based on hierarchical depth from root."""
        depths = network.depths
        if len(depths) == 0:
            return np.array([])

        min_depth = depths.min()
        max_depth = depths.max()

        if max_depth > min_depth:
            # Normalize to 0..1 where 1 is shallowest (min_depth)
            depth_normalized = 1.0 - (depths - min_depth) / (max_depth - min_depth)
        else:
            depth_normalized = np.ones_like(depths, dtype=float)

        thickness = self.min_thickness + (depth_normalized**self.power) * (
            self.max_thickness - self.min_thickness
        )
        return thickness


class DescendantThickness(BranchThicknessStrategy):
    """Thickness based on number of terminal descendants - flow-based approach."""

    def __init__(
        self,
        min_thickness: float = 0.5,
        max_thickness: float = 5.0,
        power: float = 0.5,
        mode: str = "leaves_only",
        **kwargs,
    ):
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.power = power  # For tapering curve (< 1 = gentle taper, > 1 = aggressive taper)
        self.mode = mode

    def compute_thickness(self, network: BranchNetwork) -> np.ndarray:
        """Compute thickness based on downstream terminal count."""
        num_nodes = len(network.node_ids)
        if num_nodes == 0:
            return np.array([])

        if self.mode == "leaves_only":
            descendant_counts = np.zeros(num_nodes, dtype=int)
            leaves = network.get_leaves()
            descendant_counts[leaves] = 1
        else:
            descendant_counts = np.ones(num_nodes, dtype=int)

        sorted_indices = np.argsort(network.timestamps)[::-1]
        id_to_idx = {node_id: i for i, node_id in enumerate(network.node_ids)}
        for idx in sorted_indices:
            parent_id = network.parents[idx]
            if parent_id >= 0:
                if parent_id in id_to_idx:
                    parent_idx = id_to_idx[parent_id]
                    descendant_counts[parent_idx] += descendant_counts[idx]

        min_count = descendant_counts.min()
        max_count = descendant_counts.max()

        if max_count > min_count:
            # Normalize to 0..1
            count_normalized = (descendant_counts - min_count) / (max_count - min_count)
        else:
            count_normalized = np.ones_like(descendant_counts, dtype=float)

        thickness = self.min_thickness + (count_normalized**self.power) * (
            self.max_thickness - self.min_thickness
        )
        return thickness
