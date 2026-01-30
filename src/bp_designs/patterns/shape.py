"""Shape pattern for simple geometric shapes.

Wraps polygons and other basic shapes as patterns.
"""

from __future__ import annotations

import numpy as np

from bp_designs.core.color import Color
from bp_designs.core.geometry import Canvas, Point, Polygon, Polyline
from bp_designs.core.pattern import Pattern
from bp_designs.core.renderer import RenderingContext, RenderStyle


class PointPattern(Pattern):
    """Pattern representing a single point.

    Can be absolute or relative (0-1).
    """

    def __init__(
        self,
        x: float,
        y: float,
        is_relative: bool = False,
        name: str | None = None,
        canvas: Canvas | None = None,
    ):
        super().__init__(canvas=canvas)
        self.x = x
        self.y = y
        self.is_relative = is_relative
        self._name = name

    def to_geometry(self, canvas: Canvas | None = None) -> Point:
        """Resolve to a Point geometry."""
        canvas = canvas or self.canvas
        if self.is_relative and canvas is not None:
            bounds = canvas.bounds()
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            abs_x = bounds[0] + self.x * width
            abs_y = bounds[1] + self.y * height
            return Point(x=abs_x, y=abs_y, z=None)
        return Point(x=self.x, y=self.y, z=None)

    def render(self, context: RenderingContext, style: RenderStyle | None = None, **kwargs):
        """Points don't have a standard SVG representation in our system yet."""
        pass

    def __str__(self) -> str:
        if self._name:
            return self._name
        return f"Point({'rel' if self.is_relative else 'abs'}:{self.x},{self.y})"


class ShapeStyle(RenderStyle):
    """Structured rendering parameters for shapes."""

    stroke_width: float = 0.5
    stroke_color: str | Color = "#000000"
    fill: str | Color | None = None
    stroke_linecap: str = "round"
    stroke_linejoin: str = "round"


class ShapePattern(Pattern):
    """Pattern wrapper for simple geometric shapes.

    Holds a Polygon and can convert it to geometry (as polyline).
    Supports lazy resolution against a canvas.
    """

    def __init__(
        self,
        polygon: Polygon,
        name: str | None = None,
        is_relative: bool = False,
        canvas: Canvas | None = None,
    ):
        """Initialize shape pattern.

        Args:
            polygon: The polygon defining the shape
            name: Optional descriptive name for the shape.
            is_relative: If True, coordinates are treated as 0-1 and scaled to canvas.
            canvas: Optional canvas reference.
        """
        super().__init__(canvas=canvas)
        self.polygon = polygon
        self._name = name
        self.is_relative = is_relative

    def make_relative(self, canvas: Canvas | None = None) -> ShapePattern:
        """Convert absolute coordinates to relative 0-1 coordinates based on canvas.

        Args:
            canvas: Canvas to normalize against. If None, uses self.canvas.

        Returns:
            New ShapePattern with relative coordinates.
        """
        canvas = canvas or self.canvas
        if canvas is None:
            raise ValueError("Cannot make relative without a canvas.")

        if self.is_relative:
            return self

        bounds = canvas.bounds()
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        new_coords = self.polygon.coords.copy()
        if width > 0:
            new_coords[:, 0] = (new_coords[:, 0] - bounds[0]) / width
        if height > 0:
            new_coords[:, 1] = (new_coords[:, 1] - bounds[1]) / height

        return ShapePattern(
            Polygon(coords=new_coords),
            name=self._name,
            is_relative=True,
            canvas=canvas,
        )

    def __str__(self) -> str:
        """Return human-readable string representation."""
        if self._name is not None:
            return self._name
        # Generate name from bounding box
        bounds = self.polygon.bounds()
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        return f"Shape({width:.1f}x{height:.1f})"

    def __eq__(self, other: object) -> bool:
        """Equality based on polygon coordinates."""
        if not isinstance(other, ShapePattern):
            return False
        # Compare coordinate arrays using np.array_equal
        return (
            np.array_equal(self.polygon.coords, other.polygon.coords)
            and self.is_relative == other.is_relative
        )

    def __hash__(self) -> int:
        """Hash based on polygon coordinates."""
        coords = self.polygon.coords
        if coords.size == 0:
            return hash((self.is_relative,))
        coord_tuples = tuple(tuple(map(float, point)) for point in coords)
        return hash((coord_tuples, self.is_relative))

    def to_geometry(self, canvas: Canvas | None = None) -> Polyline:
        """Convert polygon to polyline geometry.

        If is_relative is True and canvas is provided, scales coordinates.
        """
        canvas = canvas or self.canvas
        coords = self.polygon.coords

        if self.is_relative and canvas is not None:
            bounds = canvas.bounds()
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            # Scale and shift
            coords = coords.copy()
            coords[:, 0] = bounds[0] + coords[:, 0] * width
            coords[:, 1] = bounds[1] + coords[:, 1] * height

        # Ensure polygon is closed (first point == last point)
        if len(coords) > 0 and not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0:1]])

        return Polyline(polylines=[coords])

    def render(
        self,
        context: RenderingContext,
        style: ShapeStyle | None = None,
        **kwargs,
    ):
        """Render shape into the provided context."""
        # Prioritize parameters:
        # 1. Provided style object
        # 2. Stored render_params on the pattern
        # 3. kwargs passed to this method (global context)

        # Start with stored params
        params = self.render_params.copy()
        # Update with kwargs (overrides/global context)
        params.update(kwargs)

        # Merge style and params
        if style is None:
            style = ShapeStyle.from_dict(params)
        else:
            style_dict = style.model_dump()
            style_dict.update(params)
            style = ShapeStyle.from_dict(style_dict)

        # Get polygon coordinates, ensure closed
        coords = self.polygon.coords
        if len(coords) > 0 and not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0:1]])

        # Convert coordinates to list of tuples
        points = [(float(x), float(y)) for x, y in coords]

        svg_attrs = style.get_svg_attributes()

        # Draw polygon
        # Apply lighting if available
        fill = style.fill if style.fill is not None else "none"
        if context.lighting and fill != "none":
            fill = context.lighting.get_fill(fill, {"type": "global"})

        # Draw polygon
        context.add(
            context.dwg.polygon(
                points=points,
                stroke=str(style.stroke_color),
                stroke_width=style.stroke_width,
                fill=str(fill),
                stroke_linecap=style.stroke_linecap,
                stroke_linejoin=style.stroke_linejoin,
                **svg_attrs,
            )
        )
