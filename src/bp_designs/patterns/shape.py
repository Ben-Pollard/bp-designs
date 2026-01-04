"""Shape pattern for simple geometric shapes.

Wraps polygons and other basic shapes as patterns.
"""

from __future__ import annotations

import numpy as np
import svgwrite

from bp_designs.core.geometry import Polygon, Polyline
from bp_designs.core.pattern import Pattern


class ShapePattern(Pattern):
    """Pattern wrapper for simple geometric shapes.

    Holds a Polygon and can convert it to geometry (as polyline).
    """

    def __init__(self, polygon: Polygon, name: str | None = None):
        """Initialize shape pattern.

        Args:
            polygon: The polygon defining the shape
            name: Optional descriptive name for the shape. If not provided,
                a name will be generated from the bounding box.
        """
        self.polygon = polygon
        self._name = name

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
        return np.array_equal(self.polygon.coords, other.polygon.coords)

    def __hash__(self) -> int:
        """Hash based on polygon coordinates.

        Converts coordinates to tuple of tuples for hashing.
        """
        # Flatten coordinates and convert to tuple for hashing
        coords = self.polygon.coords
        if coords.size == 0:
            return hash(())
        # Convert to tuple of tuples (each point as tuple)
        coord_tuples = tuple(tuple(map(float, point)) for point in coords)
        return hash(coord_tuples)

    def to_geometry(self) -> Polyline:
        """Convert polygon to polyline geometry.

        Returns:
            Polyline containing the polygon as a closed loop.
        """
        # Ensure polygon is closed (first point == last point)
        coords = self.polygon.coords
        if len(coords) > 0 and not np.allclose(coords[0], coords[-1]):
            # Append first point to close the polygon
            coords = np.vstack([coords, coords[0:1]])

        # Return as a single polyline (closed loop)
        return Polyline(polylines=[coords])

    def to_svg(
        self,
        stroke_width: float = 0.5,
        stroke_color: str = "#000000",
        fill: str | None = None,
        stroke_linecap: str = "round",
        stroke_linejoin: str = "round",
        width: float = 800,
        height: float = 600,
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
            width: SVG canvas width in pixels
            height: SVG canvas height in pixels
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

        # Create SVG drawing
        dwg = svgwrite.Drawing(
            size=(f"{width}px", f"{height}px"),
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
