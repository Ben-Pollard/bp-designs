"""Branch network pattern package."""

from bp_designs.patterns.network.base import BranchNetwork
from bp_designs.patterns.network.color import (
    ColorStrategy,
    DepthColorStrategy,
    RandomColorStrategy,
)
from bp_designs.patterns.network.distribution import (
    ClusterDistribution,
    OrganDistributionStrategy,
    RhythmicDistribution,
    TerminalDistribution,
)
from bp_designs.patterns.network.renderer import NetworkRenderer
from bp_designs.patterns.network.thickness import (
    BranchThicknessStrategy,
    DescendantThickness,
    HierarchyThickness,
    TimestampThickness,
)

__all__ = [
    "BranchNetwork",
    "NetworkRenderer",
    "BranchThicknessStrategy",
    "TimestampThickness",
    "HierarchyThickness",
    "DescendantThickness",
    "ColorStrategy",
    "DepthColorStrategy",
    "RandomColorStrategy",
    "OrganDistributionStrategy",
    "TerminalDistribution",
    "ClusterDistribution",
    "RhythmicDistribution",
]
