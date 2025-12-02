from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from pattern import Pattern

if TYPE_CHECKING:
    from bp_designs.geometry import Geometry


class CompositePattern(Pattern):
    """A pattern created by combining other patterns.

    Implements the Pattern interface by delegating to constituent patterns.
    Can be used as input to further combinations (recursive composition).

    This is an implementation of the Composite design pattern, proving that
    the Pattern interface is properly abstract.
    """

    def __init__(self, components: list[Pattern], combinator: Callable, metadata: dict):
        """
        Args:
            components: Patterns being combined
            combinator: Function that knows how to combine them
            metadata: Information about the combination (type, parameters, etc.)
        """
        self.components = components
        self.combinator = combinator
        self.metadata = metadata

    def sample_field(self, points: np.ndarray, channel: str) -> np.ndarray:
        """Sample the composite by delegating to combinator."""
        return self.combinator(points, channel, self.components, self.metadata)

    def available_channels(self) -> dict[str, str]:
        """Union of component channels plus combination-specific ones."""
        channels = {}

        # Component channels (prefixed to avoid collisions)
        for i, component in enumerate(self.components):
            for name, desc in component.available_channels().items():
                channels[f"{name}_c{i}"] = f"{desc} (component {i})"

        # Combination-specific channels
        combo_type = self.metadata.get("type")
        if combo_type == "blend":
            mode = self.metadata.get("mode", "unknown")
            channels["blended"] = f"Blended result ({mode})"
        elif combo_type == "texture":
            channels["mask"] = "Texture mask (distance from skeleton)"

        return channels

    def to_geometry(self) -> Geometry:
        """Render by combining component geometries."""
        combo_type = self.metadata["type"]

        if combo_type == "texture":
            return self._render_texture()
        elif combo_type == "nest":
            return self._render_nested()
        elif combo_type == "blend":
            return self._render_blended()
        else:
            raise ValueError(f"Unknown combination type: {combo_type}")

    def bounds(self) -> tuple[float, float, float, float]:
        """Union of all component bounds."""
        all_bounds = [c.bounds() for c in self.components]

        xmin = min(b[0] for b in all_bounds)
        ymin = min(b[1] for b in all_bounds)
        xmax = max(b[2] for b in all_bounds)
        ymax = max(b[3] for b in all_bounds)

        return (xmin, ymin, xmax, ymax)

    def _render_texture(self) -> Geometry:
        """Combine skeleton and fill geometries."""
        # Implementation depends on specific combinator logic
        raise NotImplementedError("Implement in combinator")

    def _render_nested(self) -> Geometry:
        """Combine container and nested content geometries."""
        raise NotImplementedError("Implement in combinator")

    def _render_blended(self) -> Geometry:
        """Render field-based blend."""
        raise NotImplementedError("Implement in combinator")
