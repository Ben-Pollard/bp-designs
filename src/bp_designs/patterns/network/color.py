"""Color strategies for branching networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bp_designs.core.color import Color

if TYPE_CHECKING:
    from bp_designs.patterns.network.base import BranchNetwork


class ColorStrategy:
    """Base class for computing node colors."""

    def compute_colors(self, network: BranchNetwork) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def from_name(name: str, **kwargs) -> ColorStrategy:
        """Factory method to create strategy by name."""
        if name == "depth":
            return DepthColorStrategy(**kwargs)
        elif name == "random":
            return RandomColorStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown color strategy: {name}")


class DepthColorStrategy(ColorStrategy):
    """Color based on hierarchy depth."""

    def __init__(
        self,
        start_color: str | Color = "#4a2c2a",
        end_color: str | Color = "#2d5a27",
        **kwargs,
    ):
        self.start_color = start_color if isinstance(start_color, Color) else Color.from_hex(start_color)
        self.end_color = end_color if isinstance(end_color, Color) else Color.from_hex(end_color)

    def compute_colors(self, network: BranchNetwork) -> np.ndarray:
        depths = network.depths
        max_depth = depths.max() if len(depths) > 0 else 1

        colors = []
        for d in depths:
            t = d / max_depth
            colors.append(Color.lerp(self.start_color, self.end_color, t).to_hex())
        return np.array(colors, dtype=object)


class RandomColorStrategy(ColorStrategy):
    """Random variation around a base color."""

    def __init__(self, base_color: str | Color = "#2d5a27", variation: float = 0.1, **kwargs):
        self.base_color = base_color if isinstance(base_color, Color) else Color.from_hex(base_color)
        self.variation = variation

    def compute_colors(self, network: BranchNetwork) -> np.ndarray:
        colors = []
        rng = np.random.default_rng()
        for _ in range(len(network.node_ids)):
            colors.append(self.base_color.jitter(self.variation, rng=rng).to_hex())
        return np.array(colors, dtype=object)
