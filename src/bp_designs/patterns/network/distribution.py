"""Organ distribution strategies for branching networks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bp_designs.patterns.network.base import BranchNetwork
    from bp_designs.patterns.organs import OrganPattern


class OrganDistributionStrategy(ABC):
    """Base class for strategies that distribute organs across a network."""

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
            return TerminalDistribution()
        elif name == "cluster":
            return ClusterDistribution(**kwargs)
        elif name == "rhythmic":
            return RhythmicDistribution(**kwargs)
        else:
            raise ValueError(f"Unknown distribution strategy: {name}")


class TerminalDistribution(OrganDistributionStrategy):
    """Place organs only at terminal nodes (leaves)."""

    def generate_organs(
        self, network: BranchNetwork, organ_template: OrganPattern, **kwargs
    ) -> list[tuple[int, OrganPattern]]:
        leaves = network.get_leaves()
        return [(int(network.node_ids[leaf_idx]), organ_template) for leaf_idx in leaves]


class ClusterDistribution(OrganDistributionStrategy):
    """Place clusters of organs at terminal nodes."""

    def __init__(self, count: int = 3):
        self.count = count

    def generate_organs(
        self, network: BranchNetwork, organ_template: OrganPattern, **kwargs
    ) -> list[tuple[int, OrganPattern]]:
        leaves = network.get_leaves()
        results = []
        for leaf_idx in leaves:
            for _ in range(self.count):
                results.append((int(network.node_ids[leaf_idx]), organ_template))
        return results


class RhythmicDistribution(OrganDistributionStrategy):
    """Place organs at regular intervals along branches."""

    def __init__(self, interval: int = 5):
        self.interval = interval

    def generate_organs(
        self, network: BranchNetwork, organ_template: OrganPattern, **kwargs
    ) -> list[tuple[int, OrganPattern]]:
        results = []
        for i in range(len(network.node_ids)):
            if network.timestamps[i] % self.interval == 0:
                results.append((int(network.node_ids[i]), organ_template))
        return results
