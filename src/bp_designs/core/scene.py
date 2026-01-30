"""Scene management for layered pattern composition."""

from __future__ import annotations

import xml.dom.minidom
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import svgwrite

from .geometry import Canvas, Geometry
from .pattern import Pattern
from .renderer import RenderingContext

if TYPE_CHECKING:
    from bp_designs.core.lighting import LightingModel


@dataclass
class Layer:
    """A single layer in a scene."""

    name: str
    pattern: Pattern
    params: dict[str, Any] = field(default_factory=dict)
    visible: bool = True


class Scene(Pattern):
    """A composite pattern that manages multiple layers and a canvas."""

    def __init__(self, canvas: Canvas, render_params: dict[str, Any] | None = None):
        super().__init__(canvas=canvas, render_params=render_params or {})
        self.canvas = canvas
        self.layers: list[Layer] = []

    def add_layer(self, name: str, pattern: Pattern, **params):
        """Add a new layer to the scene."""
        self.layers.append(Layer(name=name, pattern=pattern, params=params))

    def render(self, context: RenderingContext, **kwargs):
        """Render all visible layers in the scene."""
        # Render background from canvas
        if self.canvas.background_color:
            xmin, ymin, xmax, ymax = self.canvas.bounds()
            fill = self.canvas.background_color
            if context.lighting:
                fill = context.lighting.get_fill(fill, {"type": "background"})
            context.add(
                context.dwg.rect(
                    insert=(xmin, ymin),
                    size=(xmax - xmin, ymax - ymin),
                    fill=str(fill),
                )
            )

        # Render layers in order
        for layer in self.layers:
            if layer.visible:
                # Merge layer params with global kwargs, layer params take precedence
                render_params = {**kwargs, **layer.params}

                # Extract SVG group attributes (like transform) from render_params
                group_attrs = {}
                if "transform" in render_params:
                    group_attrs["transform"] = render_params.pop("transform")

                context.push_group(layer.name, **group_attrs)
                layer.pattern.render(context, **render_params)
                context.pop_group()

    def to_geometry(self, canvas: Canvas | None = None) -> Geometry:
        """Convert all layers to geometry (not fully implemented for all types)."""
        # For now, just return the canvas bounds as a polygon
        return self.canvas

    def to_svg(
        self,
        width: str | float = "100%",
        height: str | float = "100%",
        padding: float = 20,
        pretty: bool = True,
        **kwargs,
    ) -> str:
        """Render scene to a standalone SVG string."""
        xmin, ymin, xmax, ymax = self.canvas.bounds()
        xmin -= padding
        ymin -= padding
        xmax += padding
        ymax += padding
        view_width = xmax - xmin
        view_height = ymax - ymin

        def format_size(s):
            return s if isinstance(s, str) else f"{s}px"

        dwg = svgwrite.Drawing(
            size=(format_size(width), format_size(height)),
            viewBox=f"{xmin} {ymin} {view_width} {view_height}",
        )

        lighting: LightingModel | None = kwargs.get("lighting")
        context = RenderingContext(dwg, lighting=lighting)
        self.render(context, **kwargs)

        svg_string = dwg.tostring()

        if pretty:
            try:
                dom = xml.dom.minidom.parseString(svg_string)
                return dom.toprettyxml()
            except Exception:
                # Fallback to raw string if pretty printing fails
                return svg_string

        return svg_string

    def __str__(self) -> str:
        return f"Scene(layers={len(self.layers)}, canvas={self.canvas})"
