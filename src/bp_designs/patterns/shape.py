"""Shape pattern for simple geometric shapes.

Wraps polygons and other basic shapes as patterns.
"""

from __future__ import annotations

import numpy as np
import svgwrite

from bp_designs.core.geometry import Canvas, Point, Polygon, Polyline
from bp_designs.core.pattern import Pattern


class PointPattern(Pattern):
    """Pattern representing a single point.

    Can be absolute or relative (0-1).
    """

    def __init__(self, x: float, y: float, is_relative: bool = False, name: str | None = None):
        self.x = x
        self.y = y
        self.is_relative = is_relative
        self._name = name

    def to_geometry(self, canvas: Canvas | None = None) -> Point:
        """Resolve to a Point geometry."""
        if self.is_relative and canvas is not None:
            bounds = canvas.bounds()
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            abs_x = bounds[0] + self.x * width
            abs_y = bounds[1] + self.y * height
            return Point(x=abs_x, y=abs_y, z=None)
        return Point(x=self.x, y=self.y, z=None)

    def to_svg(self, **kwargs) -> str:
        """Points don't have a standard SVG representation in our system yet."""
        return ""

    def __str__(self) -> str:
        if self._name:
            return self._name
        return f"Point({'rel' if self.is_relative else 'abs'}:{self.x},{self.y})"


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
    ):
        """Initialize shape pattern.

        Args:
            polygon: The polygon defining the shape
            name: Optional descriptive name for the shape.
            is_relative: If True, coordinates are treated as 0-1 and scaled to canvas.
        """
        self.polygon = polygon
        self._name = name
        self.is_relative = is_relative

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
        return np.array_equal(self.polygon.coords, other.polygon.coords) and self.is_relative == other.is_relative

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

    def to_svg(
        self,
        stroke_width: float = 0.5,
        stroke_color: str = "#000000",
        fill: str | None = None,
        stroke_linecap: str = "round",
        stroke_linejoin: str = "round",
        width: str | float = "100%",
        height: str | float = "100%",
        padding: float = 20,
        background: str | None = None,
    ) -> str:
        """Render shape as SVG.

        Args:
            stroke_width: Line width in SVG units
            stroke_color: Stroke color (any valid SVG color)
            fill: Fill color (or None for no fill)
            stroke_linecap: SVG linecap style ('round', 'butt', 'square')
            stroke_linejoin: SVG linejoin style ('round', 'miter', 'bevel')
            width: SVG canvas width (default '100%' for responsive)
            height: SVG canvas height (default '100%' for responsive)
            padding: Padding around shape in SVG units
            background: Background color (or None for transparent)

        Returns:
            SVG string
        """
        # Get polygon coordinates, ensure closed
        coords = self.polygon.coords
        if len(coords) > 0 and not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0:1]])

        # Compute bounds with padding
        xmin, ymin, xmax, ymax = self.polygon.bounds()
        xmin -= padding
        ymin -= padding
        xmax += padding
        ymax += padding
        view_width = xmax - xmin
        view_height = ymax - ymin

        # Format size for svgwrite
        def format_size(s):
            if isinstance(s, str):
                return s
            return f"{s}px"

        # Create SVG drawing
        dwg = svgwrite.Drawing(
            size=(format_size(width), format_size(height)),
            viewBox=f"{xmin} {ymin} {view_width} {view_height}",
        )

        # Add background if specified
        if background is not None:
            dwg.add(dwg.rect(
                insert=(xmin, ymin),
                size=(view_width, view_height),
                fill=background,
            ))

        # Convert coordinates to list of tuples
        points = [(float(x), float(y)) for x, y in coords]

        # Draw polygon
        dwg.add(dwg.polygon(
            points=points,
            stroke=stroke_color,
            stroke_width=stroke_width,
            fill=fill if fill is not None else "none",
            stroke_linecap=stroke_linecap,
            stroke_linejoin=stroke_linejoin,
        ))

        return dwg.tostring()
