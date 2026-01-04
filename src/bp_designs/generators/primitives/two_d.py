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


class RegularPolygon(Primitive2D):
    """Generate regular polygon shapes.

    Creates regular polygons with specified number of sides, radius, and rotation.
    """

    def __init__(
        self,
        sides: int,
        radius: float,
        center: tuple[float, float] = (0.0, 0.0),
        rotation: float = 0.0,
        canvas: Canvas | None = None,
        name: str | None = None,
    ):
        """Initialize regular polygon.

        Args:
            sides: Number of sides (minimum 3)
            radius: Distance from center to vertices
            center: (x, y) center position
            rotation: Rotation angle in radians (0 = first vertex at angle 0)
            canvas: Optional canvas to bound the polygon within. If provided,
                the polygon will be scaled to fit within canvas (centered automatically).
            name: Optional descriptive name for the polygon pattern.
        """
        if sides < 3:
            raise ValueError(f"Regular polygon must have at least 3 sides, got {sides}")
        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")

        self.sides = sides
        self.radius = radius
        self.center = center
        self.rotation = rotation
        self.canvas = canvas
        self.name = name

    @classmethod
    def from_bbox(
        cls,
        sides: int,
        bbox: tuple[float, float, float, float] | list[float],
        canvas: Canvas | None = None,
        name: str | None = None,
    ) -> RegularPolygon:
        """Create a RegularPolygon instance from bounding box.

        The polygon will be inscribed within the bounding box (largest regular polygon
        that fits within the rectangle).

        Args:
            sides: Number of sides
            bbox: Bounding box as (x0, y0, x1, y1) where (x0,y0) is top-left
                and (x1,y1) is bottom-right (like PIL).
            canvas: Optional canvas to bound the polygon within.
            name: Optional descriptive name for the polygon pattern.

        Returns:
            RegularPolygon instance
        """
        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        # Radius is half of the smaller dimension (inscribed circle)
        radius = min(width, height) / 2

        return cls(
            sides=sides,
            radius=radius,
            center=(cx, cy),
            rotation=0.0,
            canvas=canvas,
            name=name,
        )

    @classmethod
    def from_width_height(
        cls,
        sides: int,
        width: float,
        height: float,
        canvas: Canvas | None = None,
        name: str | None = None,
    ) -> RegularPolygon:
        """Create a RegularPolygon instance from width and height.

        Creates a bounding box centered at origin (0,0) with given dimensions,
        then creates polygon inscribed within it.

        Args:
            sides: Number of sides
            width: Width of bounding box
            height: Height of bounding box
            canvas: Optional canvas to bound the polygon within.
            name: Optional descriptive name for the polygon pattern.

        Returns:
            RegularPolygon instance
        """
        x0 = -width / 2
        y0 = -height / 2
        x1 = width / 2
        y1 = height / 2
        return cls.from_bbox(sides=sides, bbox=(x0, y0, x1, y1), canvas=canvas, name=name)

    def generate_pattern(self, **kwargs) -> ShapePattern:
        """Generate regular polygon.

        Returns:
            ShapePattern containing a Polygon representing the regular polygon.
        """
        sides = self.sides
        radius = self.radius
        cx, cy = self.center
        rotation = self.rotation

        # If canvas is provided, adjust position and scale to fit
        if self.canvas is not None:
            # Scale polygon proportionally to fit within canvas if needed
            canvas_bounds = self.canvas.bounds()
            canvas_width = canvas_bounds[2] - canvas_bounds[0]
            canvas_height = canvas_bounds[3] - canvas_bounds[1]

            # Current bounding box size (circumscribed circle diameter = 2 * radius)
            current_size = 2 * radius
            scale = min(canvas_width / current_size, canvas_height / current_size, 1.0)
            radius = radius * scale

            # Center in canvas (override provided center)
            cx = (canvas_bounds[0] + canvas_bounds[2]) / 2
            cy = (canvas_bounds[1] + canvas_bounds[3]) / 2

        # Generate regular polygon vertices
        # Start at angle 0 + rotation, evenly spaced around circle
        angles = np.linspace(0, 2 * np.pi, sides, endpoint=False) + rotation
        x = cx + radius * np.cos(angles)
        y = cy + radius * np.sin(angles)

        # Close the polygon by repeating first point
        coords = np.column_stack([x, y])
        # Ensure closure (first point == last point)
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0:1]])

        polygon = Polygon(coords=coords)
        return ShapePattern(polygon, name=self.name)
