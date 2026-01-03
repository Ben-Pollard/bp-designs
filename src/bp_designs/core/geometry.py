from abc import ABC
from dataclasses import dataclass

import numpy as np
import svgwrite


class Geometry(ABC):
    """
    Base class for geometries. Servers as a data layer to translate between
    internal and library geometry representations.
    """

    dim: int
    attrs: dict


@dataclass
class Point(Geometry):
    x: int
    y: int
    z: int | None

    def __str__(self) -> str:
        """Return concise string representation."""
        if self.z is None:
            return f"Point({self.x},{self.y})"
        else:
            return f"Point({self.x},{self.y},{self.z})"


@dataclass
class PointSet(Geometry):
    points: np.ndarray  # shape (N, D)
    channels: dict[str, np.ndarray]  # each shape (N, k)


@dataclass
class Polygon(Geometry):
    """2D polygon defined by sequence of points."""

    coords: np.ndarray  # shape (N, 2) - polygon vertices in order

    def __post_init__(self):
        """Validate shape."""
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError(f"coords must be (N, 2), got {self.coords.shape}")

    def __str__(self) -> str:
        """Return concise string representation."""
        bounds = self.bounds()
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        return f"Polygon({width:.1f}x{height:.1f})"

    def bounds(self) -> tuple[float, float, float, float]:
        """Return bounding box (xmin, ymin, xmax, ymax)."""
        if len(self.coords) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        xmin, ymin = self.coords.min(axis=0)
        xmax, ymax = self.coords.max(axis=0)
        return (float(xmin), float(ymin), float(xmax), float(ymax))


@dataclass
class Canvas(Polygon):

    def __str__(self) -> str:
        """Return concise string representation."""
        bounds = self.bounds()
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        return f"Canvas({width:.0f}x{height:.0f})"

    @classmethod
    def from_width_height(cls, width: int, height: int):
        coords = np.array([
            [0.0, 0.0],
            [width, 0.0],
            [width, height],
            [0.0, height],
            [0.0, 0.0],  # Close the polygon
        ])
        return cls(coords=coords)

    @classmethod
    def from_size(cls, size: int):

        return cls.from_width_height(size, size)


@dataclass
class Mesh(Geometry):
    pass


@dataclass
class Polyline(Geometry):
    polylines: list[np.ndarray]

    def to_svg(
        self,
        width: float = 100,
        height: float = 100,
        stroke_width: float = 0.5,
        stroke_color: str = "#000000",
        background: str | None = "#ffffff",
    ) -> str:
        """Convert geometry to SVG string.

        Args:
            geometry: List of polylines to render
            width: SVG width in mm
            height: SVG height in mm
            stroke_width: Line width in mm
            stroke_color: Stroke color (hex)
            background: Background color (hex) or None for transparent

        Returns:
            SVG string
        """
        # Create SVG with viewBox for proper scaling
        dwg = svgwrite.Drawing(size=(f"{width}mm", f"{height}mm"), viewBox=f"0 0 {width} {height}")

        # Add background if specified
        if background:
            dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill=background))

        # Add each polyline
        for polyline in self.polylines:
            if len(polyline) < 2:
                continue

            points = [(float(p[0]), float(p[1])) for p in polyline]
            dwg.add(
                dwg.polyline(
                    points=points,
                    stroke=stroke_color,
                    stroke_width=stroke_width,
                    fill="none",
                    stroke_linecap="round",
                    stroke_linejoin="round",
                )
            )

        return dwg.tostring()

    def render_svg(
        self,
        width: float = 100,
        height: float = 100,
        stroke_width: float = 0.5,
        stroke_color: str = "#000000",
        background: str | None = "#ffffff",
    ):
        """Render geometry as SVG for Jupyter display.

        This function returns an object that Jupyter will automatically
        display as an SVG image.

        Args:
            geometry: List of polylines to render
            width: SVG width in mm
            height: SVG height in mm
            stroke_width: Line width in mm
            stroke_color: Stroke color (hex)
            background: Background color (hex) or None for transparent

        Returns:
            IPython.display.SVG object (if in Jupyter) or SVG string
        """
        svg_string = self.to_svg(
            width=width,
            height=height,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background=background,
        )

        # Try to use IPython display if available
        try:
            from IPython.display import SVG

            return SVG(data=svg_string)
        except ImportError:
            # Not in Jupyter, just return string
            return svg_string
