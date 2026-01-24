"""Base Pattern interface for composable generative patterns.

Semantic representation of output of Generators.

Implementations should wrap battle-tested library code for common patterns,
e.g. trees, scalar fields, vector fields

All generative patterns implement this interface, enabling:
1. Semantic queries
2. Geometry export - convert to renderable/manufacturable format
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .geometry import Canvas, Geometry
    from .renderer import RenderingContext, RenderStyle
    from .scene import Layer


@dataclass(kw_only=True)
class Pattern(ABC):
    """Interface for patterns that can be spatially queried and rendered.

    A Pattern:
    - Can be rendered to geometry (to_geometry)
    - Has a string representation for display and identification

    Patterns maintain their own internal semantic structures (e.g.,
    BranchNetwork maintains parents, depths, branch_ids as separate arrays).
    The Pattern interface is how they expose themselves to the outside world.
    """

    # Default rendering parameters for this pattern instance
    render_params: dict[str, Any] = field(default_factory=dict)

    # Coordinate system reference
    canvas: Canvas | None = None

    @abstractmethod
    def to_geometry(self, canvas: Canvas | None = None) -> Geometry:
        """Convert pattern to renderable geometry.

        Args:
            canvas: Optional canvas to resolve relative coordinates against.
                If provided, the pattern will be scaled/positioned to fit.
        """
        pass

    @abstractmethod
    def render(self, context: RenderingContext, style: RenderStyle | None = None, **kwargs):
        """Render pattern into the provided context.

        Args:
            context: Rendering context (e.g. SVG drawing wrapper)
            style: Structured rendering parameters.
            **kwargs: Additional rendering parameters (overrides style).
        """
        pass

    def to_layers(self) -> list[Layer]:
        """Decompose pattern into multiple layers for rendering.

        Default implementation returns the pattern itself as a single layer.
        """
        from .scene import Layer

        return [Layer(name=getattr(self, "name", "pattern"), pattern=self)]

    def to_svg(self, **kwargs) -> str:
        """Definitive entry point for rendering a pattern to a standalone SVG string.

        This method uses the Scene and RenderingContext system to ensure a
        consistent rendering flow.

        Args:
            **kwargs: Overrides for render_params and Scene settings.
        """
        from .scene import Scene

        # Merge instance render_params with overrides
        params = {**self.render_params, **kwargs}

        # Try to get canvas from self if it exists, otherwise use default
        canvas = kwargs.pop("canvas", getattr(self, "canvas", None))
        if canvas is None:
            raise ValueError(
                f"Pattern {self} has no canvas. A canvas must be supplied for rendering. "
                "Pass it to to_svg(canvas=...) or ensure the pattern has a .canvas attribute."
            )

        # Scene-level parameters should be handled here and not passed to layers
        bg_color = params.pop("background_color", None)
        if bg_color:
            canvas.background_color = bg_color

        scene = Scene(canvas)

        # Use to_layers to decompose the pattern if supported
        layers = self.to_layers()
        for layer in layers:
            # Merge layer-specific params with global overrides
            layer_params = {**layer.params, **params}
            scene.add_layer(layer.name, layer.pattern, **layer_params)

        return scene.to_svg(**params)

    @abstractmethod
    def __str__(self) -> str:
        """Return human-readable string representation of the pattern."""
        pass
