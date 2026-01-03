"""2D primitive shape generators.

Generates basic geometric shapes as ShapePattern instances.
"""

from __future__ import annotations

import numpy as np

from bp_designs.core.generator import Generator
from bp_designs.core.geometry import Canvas, Polygon
from bp_designs.patterns.shape import ShapePattern


class Primitive2D(Generator):
    """Generate 2D primitive shapes as patterns.

    Supports circles, rectangles, regular polygons, and arbitrary polygons.

    This only needs to know about how to create the semantic information for a shape

    ShapePattern holds the semantic information and knows how to create a Geometry from it.

    For something simple like a circle that could just be passing the centre Point and radius
    through the Pattern
    """
    pass


class Oval(Primitive2D):
    """Generate oval (ellipse) shapes.

    The oval is approximated as a polygon with many segments.
    Primary construction is via bounding box (like PIL).
    """

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        canvas: Canvas | None = None,
        name: str | None = None,
    ):
        """Initialize oval from bounding box.

        Args:
            bbox: Bounding box as (x0, y0, x1, y1) where (x0,y0) is top-left
                and (x1,y1) is bottom-right (like PIL).
            canvas: Optional canvas to bound the oval within. If provided,
                the oval will be scaled to fit within canvas (centered automatically).
            name: Optional descriptive name for the oval pattern.
        """
        self.bbox = bbox
        self.canvas = canvas
        self.name = name

    @classmethod
    def from_bbox(
        cls,
        bbox: tuple[float, float, float, float] | list[float],
        canvas: Canvas | None = None,
        name: str | None = None,
    ) -> Oval:
        """Create an Oval instance from bounding box.

        Primary constructor.
        """
        return cls(bbox=tuple(bbox), canvas=canvas, name=name)

    @classmethod
    def from_width_height(
        cls,
        width: float,
        height: float,
        canvas: Canvas | None = None,
        name: str | None = None,
    ) -> Oval:
        """Create an Oval instance from width and height.

        Creates a bounding box centered at origin (0,0) with given dimensions.
        """
        x0 = -width / 2
        y0 = -height / 2
        x1 = width / 2
        y1 = height / 2
        return cls.from_bbox(bbox=(x0, y0, x1, y1), canvas=canvas, name=name)

    def generate_pattern(self, **kwargs) -> ShapePattern:
        """Generate oval as a polygon.

        Returns:
            ShapePattern containing a Polygon approximating the oval.
        """
        x0, y0, x1, y1 = self.bbox
        width = x1 - x0
        height = y1 - y0
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        # If canvas is provided, adjust position and scale to fit
        if self.canvas is not None:
            # Scale oval proportionally to fit within canvas if needed
            canvas_bounds = self.canvas.bounds()
            canvas_width = canvas_bounds[2] - canvas_bounds[0]
            canvas_height = canvas_bounds[3] - canvas_bounds[1]

            scale = min(canvas_width / width, canvas_height / height, 1.0)
            width = width * scale
            height = height * scale

            # Center in canvas (override provided center)
            cx = (canvas_bounds[0] + canvas_bounds[2]) / 2
            cy = (canvas_bounds[1] + canvas_bounds[3]) / 2

        # Generate oval polygon points (ellipse approximation)
        num_segments = 64  # Enough for smooth appearance
        theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)

        # Ellipse parametric equations
        x = cx + (width / 2) * np.cos(theta)
        y = cy + (height / 2) * np.sin(theta)

        # Close the polygon by repeating first point
        coords = np.column_stack([x, y])
        # Ensure closure (first point == last point)
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0:1]])

        polygon = Polygon(coords=coords)
        return ShapePattern(polygon, name=self.name)
