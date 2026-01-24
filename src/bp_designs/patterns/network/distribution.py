"""Organ distribution strategies for branching networks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bp_designs.patterns.network.base import BranchNetwork
    from bp_designs.patterns.organs import OrganPattern


class OrganDistributionStrategy(ABC):
    """Base class for strategies that distribute organs across a network."""

    def __init__(self, min_depth_ratio: float = 0.0, random_chance: float = 1.0, seed: int | None = None):
        """Initialize distribution strategy.

        Args:
            min_depth_ratio: Only place organs on nodes with depth >= max_depth * min_depth_ratio.
            random_chance: Probability (0-1) of placing an organ at a valid node.
            seed: Random seed for thinning.
        """
        self.min_depth_ratio = min_depth_ratio
        self.random_chance = random_chance
        self.rng = np.random.default_rng(seed)

    def _should_place(self, network: BranchNetwork, node_idx: int) -> bool:
        """Check if an organ should be placed at the given node based on common filters."""
        if self.min_depth_ratio > 0:
            max_depth = network.max_depth
            if max_depth > 0:
                if network.depths[node_idx] < max_depth * self.min_depth_ratio:
                    return False

        if self.random_chance < 1.0:
            if self.rng.random() > self.random_chance:
                return False

        return True

    @abstractmethod
    def generate_organs(
        self, network: BranchNetwork, organ_template: OrganPattern, **kwargs
    ) -> list[tuple[int, OrganPattern]]:
        """Generate organ instances for specific nodes.

        Returns:
            List of (node_id, organ_instance) tuples.
        """
        pass

    @staticmethod
    def from_name(name: str | OrganDistributionStrategy, **kwargs) -> OrganDistributionStrategy:
        """Factory method to create strategy by name or return if already a strategy."""
        if isinstance(name, OrganDistributionStrategy):
            return name
        if name == "terminal":
            return TerminalDistribution(**kwargs)
        elif name == "cluster":
            return ClusterDistribution(**kwargs)
        elif name == "rhythmic":
            return RhythmicDistribution(**kwargs)
        else:
            raise ValueError(f"Unknown distribution strategy: {name}")


class TerminalDistribution(OrganDistributionStrategy):
    """Place organs only at terminal nodes (leaves)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_organs(
        self, network: BranchNetwork, organ_template: OrganPattern, **kwargs
    ) -> list[tuple[int, OrganPattern]]:
        leaves = network.get_leaves()
        return [
            (int(network.node_ids[leaf_idx]), organ_template)
            for leaf_idx in leaves
            if self._should_place(network, leaf_idx)
        ]


class ClusterDistribution(OrganDistributionStrategy):
    """Place clusters of organs at terminal nodes."""

    def __init__(self, count: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.count = count

    def generate_organs(
        self, network: BranchNetwork, organ_template: OrganPattern, **kwargs
    ) -> list[tuple[int, OrganPattern]]:
        leaves = network.get_leaves()
        results = []
        for leaf_idx in leaves:
            if self._should_place(network, leaf_idx):
                for _ in range(self.count):
                    results.append((int(network.node_ids[leaf_idx]), organ_template))
        return results


class RhythmicDistribution(OrganDistributionStrategy):
    """Place organs at regular intervals along branches."""

    def __init__(self, interval: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.interval = interval

    def generate_organs(
        self, network: BranchNetwork, organ_template: OrganPattern, **kwargs
    ) -> list[tuple[int, OrganPattern]]:
        results = []
        # Use depths for rhythmic distribution to avoid issues with subdivided timestamps
        # and ensure more consistent spacing along branches.
        depths = network.depths
        for i in range(len(network.node_ids)):
            if depths[i] > 0 and depths[i] % self.interval == 0:
                if self._should_place(network, i):
                    results.append((int(network.node_ids[i]), organ_template))
        return results
