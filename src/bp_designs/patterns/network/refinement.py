"""Refinement strategies for branching networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bp_designs.patterns.network.base import BranchNetwork


@dataclass
class NetworkRefinementStrategy:
    """Encapsulates a sequence of refinement steps.

    The strategy is defined by the options provided. If an option is set,
    the corresponding refinement step is applied in a fixed order:
    1. Decimate (if decimate_min_distance is set)
    2. Relocate (if relocate_alpha is set)
    3. Subdivide (if subdivide is True)
    """

    decimate_min_distance: float | None = None
    relocate_alpha: float | None = None
    subdivide: bool = False
    relocate_iterations: int = 1
    relocate_fix_roots: bool = True
    relocate_fix_leaves: bool = True

    def apply(self, network: BranchNetwork) -> BranchNetwork:
        """Apply enabled refinement steps in sequence."""
        if self.decimate_min_distance is not None:
            network = network.decimate(min_distance=self.decimate_min_distance)
        if self.relocate_alpha is not None:
            network = network.relocate(
                alpha=self.relocate_alpha,
                iterations=self.relocate_iterations,
                fix_roots=self.relocate_fix_roots,
                fix_leaves=self.relocate_fix_leaves,
            )
        if self.subdivide:
            network = network.subdivide()
        return network
