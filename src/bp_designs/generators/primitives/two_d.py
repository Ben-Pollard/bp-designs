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
        is_relative: bool = False,
    ):
        """Initialize oval from bounding box.

        Args:
            bbox: Bounding box as (x0, y0, x1, y1) where (x0,y0) is top-left
                and (x1,y1) is bottom-right (like PIL).
            canvas: Optional canvas to bound the oval within. If provided,
                the oval will be scaled to fit within canvas (centered automatically).
            name: Optional descriptive name for the oval pattern.
            is_relative: Whether the bbox is in relative 0-1 coordinates.
        """
        self.bbox = bbox
        self.canvas = canvas
        self.name = name
        self.is_relative = is_relative

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
        is_relative = all(0.0 <= v <= 1.0 for v in bbox)
        return cls(bbox=tuple(bbox), canvas=canvas, name=name, is_relative=is_relative)

    @classmethod
    def from_width_height(
        cls,
        width: float,
        height: float,
        canvas: Canvas | None = None,
        name: str | None = None,
    ) -> Oval:
        """Create an Oval instance from width and height.

        If canvas is provided, centers the oval on the canvas.
        Otherwise, centers at origin (0,0).
        """
        is_relative = False
        if canvas is not None:
            if width <= 1.0 and height <= 1.0:
                is_relative = True
                cx, cy = 0.5, 0.5
            else:
                bounds = canvas.bounds()
                cx = (bounds[0] + bounds[2]) / 2
                cy = (bounds[1] + bounds[3]) / 2
        else:
            cx, cy = 0.0, 0.0

        x0 = cx - width / 2
        y0 = cy - height / 2
        x1 = cx + width / 2
        y1 = cy + height / 2

        return cls(bbox=(x0, y0, x1, y1), canvas=canvas, name=name, is_relative=is_relative)

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
        return ShapePattern(
            polygon, name=self.name, is_relative=self.is_relative, canvas=self.canvas
        )


class RegularPolygon(Primitive2D):
    """Generate regular polygon shapes.

    Creates regular polygons with specified number of sides, radius, and rotation.
    """

    def __init__(
        self,
        sides: int,
        radius: float | tuple[float, float],
        center: tuple[float, float] = (0.0, 0.0),
        rotation: float = 0.0,
        canvas: Canvas | None = None,
        name: str | None = None,
        is_relative: bool = False,
    ):
        """Initialize regular polygon.

        Args:
            sides: Number of sides (minimum 3)
            radius: Distance from center to vertices. If tuple (rx, ry), creates an ellipse-like polygon.
            center: (x, y) center position
            rotation: Rotation angle in radians (0 = first vertex at angle 0)
            canvas: Optional canvas to bound the polygon within. If provided,
                the polygon will be scaled to fit within canvas (centered automatically).
            name: Optional descriptive name for the polygon pattern.
            is_relative: Whether coordinates are in relative 0-1 space.
        """
        if sides < 3:
            raise ValueError(f"Regular polygon must have at least 3 sides, got {sides}")

        self.sides = sides
        self.radius = radius
        self.center = center
        self.rotation = rotation
        self.canvas = canvas
        self.name = name
        self.is_relative = is_relative

    @classmethod
    def from_bbox(
        cls,
        sides: int,
        bbox: tuple[float, float, float, float] | list[float],
        canvas: Canvas | None = None,
        name: str | None = None,
    ) -> RegularPolygon:
        """Create a RegularPolygon instance from bounding box.

        The polygon will be inscribed within the bounding box.

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

        # For regular polygons, we usually want a single radius.
        # But we can support non-uniform scaling by passing a tuple.
        radius = (width / 2, height / 2)

        is_relative = all(0.0 <= v <= 1.0 for v in bbox)

        return cls(
            sides=sides,
            radius=radius,
            center=(cx, cy),
            rotation=0.0,
            canvas=canvas,
            name=name,
            is_relative=is_relative,
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

        If canvas is provided, centers the polygon on the canvas.
        Otherwise, centers at origin (0,0).

        Args:
            sides: Number of sides
            width: Width of bounding box
            height: Height of bounding box
            canvas: Optional canvas to bound the polygon within.
            name: Optional descriptive name for the polygon pattern.

        Returns:
            RegularPolygon instance
        """
        if canvas is not None:
            if width <= 1.0 and height <= 1.0:
                is_relative = True
                cx, cy = 0.5, 0.5
            else:
                is_relative = False
                bounds = canvas.bounds()
                cx = (bounds[0] + bounds[2]) / 2
                cy = (bounds[1] + bounds[3]) / 2
        else:
            is_relative = False
            cx, cy = 0.0, 0.0

        radius = (width / 2, height / 2)

        return cls(
            sides=sides,
            radius=radius,
            center=(cx, cy),
            rotation=0.0,
            canvas=canvas,
            name=name,
            is_relative=is_relative,
        )

    def generate_pattern(self, **kwargs) -> ShapePattern:
        """Generate regular polygon.

        Returns:
            ShapePattern containing a Polygon representing the regular polygon.
        """
        sides = self.sides
        if isinstance(self.radius, tuple):
            rx, ry = self.radius
        else:
            rx = ry = self.radius
        cx, cy = self.center
        rotation = self.rotation

        # Generate regular polygon vertices
        # Start at angle 0 + rotation, evenly spaced around circle
        angles = np.linspace(0, 2 * np.pi, sides, endpoint=False) + rotation
        x = cx + rx * np.cos(angles)
        y = cy + ry * np.sin(angles)

        # Close the polygon by repeating first point
        coords = np.column_stack([x, y])
        # Ensure closure (first point == last point)
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0:1]])

        polygon = Polygon(coords=coords)
        return ShapePattern(
            polygon, name=self.name, is_relative=self.is_relative, canvas=self.canvas
        )


class Rectangle(RegularPolygon):
    """Generate rectangular shapes.

    A rectangle is a 4-sided regular polygon rotated by 45 degrees.
    """

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        canvas: Canvas | None = None,
        name: str | None = None,
        is_relative: bool = False,
    ):
        """Initialize rectangle from bounding box.

        Args:
            bbox: Bounding box as (x0, y0, x1, y1)
            canvas: Optional canvas
            name: Optional name
            is_relative: Whether bbox is in relative coordinates
        """
        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        # For a rectangle, we want the vertices at the corners of the bbox.
        # A regular polygon with 4 sides has vertices at (r, 0), (0, r), (-r, 0), (0, -r)
        # if rotation is 0.
        # To get a rectangle aligned with axes, we rotate by 45 degrees (pi/4).
        # The radius needs to be adjusted so that the corners hit the bbox.
        # Corner distance from center is sqrt((w/2)^2 + (h/2)^2)
        rx = width / 2
        ry = height / 2

        # We use the base RegularPolygon with 4 sides and pi/4 rotation.
        # However, RegularPolygon uses rx*cos(theta), ry*sin(theta).
        # For a rectangle, we actually want:
        # x = cx +/- width/2
        # y = cy +/- height/2
        # This is NOT what RegularPolygon(sides=4, rotation=pi/4) gives with rx, ry.
        # It gives a diamond if rx != ry.
        # Wait, if rx == ry it's a square.
        # If we want a rectangle, we can just define the 4 points directly or
        # override generate_pattern.

        super().__init__(
            sides=4,
            radius=(rx, ry),
            center=(cx, cy),
            rotation=np.pi / 4,
            canvas=canvas,
            name=name,
            is_relative=is_relative,
        )

    @classmethod
    def from_canvas(cls, canvas: Canvas, name: str | None = None) -> Rectangle:
        """Create a rectangle that matches the canvas bounds."""
        return cls(bbox=canvas.bounds(), canvas=canvas, name=name, is_relative=False)

    def generate_pattern(self, **kwargs) -> ShapePattern:
        """Generate rectangle as a 4-point polygon."""
        # Overriding to ensure it's a perfect axis-aligned rectangle
        x0, y0, x1, y1 = self.center[0] - self.radius[0], self.center[1] - self.radius[1], \
                         self.center[0] + self.radius[0], self.center[1] + self.radius[1]

        coords = np.array([
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
            [x0, y0]
        ])

        polygon = Polygon(coords=coords)
        return ShapePattern(
            polygon, name=self.name, is_relative=self.is_relative, canvas=self.canvas
        )
