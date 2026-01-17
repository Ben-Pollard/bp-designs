from abc import ABC
from dataclasses import dataclass

import numpy as np


class Geometry(ABC):
    """
    Base class for geometries. Servers as a data layer to translate between
    internal and library geometry representations.
    """

    dim: int
    attrs: dict


@dataclass
class Point(Geometry):
    x: float
    y: float
    z: float | None

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
    background_color: str | None = None

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

