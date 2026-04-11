from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from bp_designs.core.geometry import Canvas, Geometry, Polyline
from bp_designs.core.pattern import Pattern
from bp_designs.core.renderer import RenderStyle

if TYPE_CHECKING:
    from bp_designs.core.renderer import RenderingContext


class WidthStrategy(ABC):
    """Abstract base class for streamline width strategies."""

    @abstractmethod
    def get_width(self, streamline: np.ndarray, index: int, magnitudes: np.ndarray | None = None) -> float:
        pass


class ConstantWidth(WidthStrategy):
    def __init__(self, width: float = 1.0):
        self.width = width

    def get_width(self, streamline: np.ndarray, index: int, magnitudes: np.ndarray | None = None) -> float:
        return self.width


class TaperedWidth(WidthStrategy):
    def __init__(self, min_width: float = 0.5, max_width: float = 2.0):
        self.min_width = min_width
        self.max_width = max_width

    def get_width(self, streamline: np.ndarray, index: int, magnitudes: np.ndarray | None = None) -> float:
        progress = index / (len(streamline) - 1)
        return self.max_width * (1.0 - progress) + self.min_width * progress


class ColorStrategy(ABC):
    """Abstract base class for streamline color strategies."""

    @abstractmethod
    def get_color(self, streamline: np.ndarray, index: int, magnitudes: np.ndarray | None = None) -> str:
        pass


class ConstantColor(ColorStrategy):
    def __init__(self, color: str = "#000000"):
        self.color = color

    def get_color(self, streamline: np.ndarray, index: int, magnitudes: np.ndarray | None = None) -> str:
        return self.color


class AngleColor(ColorStrategy):
    def get_color(self, streamline: np.ndarray, index: int, magnitudes: np.ndarray | None = None) -> str:
        p1, p2 = streamline[index], streamline[index + 1]
        v = p2 - p1
        angle = np.arctan2(v[1], v[0])
        hue = (angle + np.pi) / (2 * np.pi)
        from bp_designs.core.color import Color

        return Color.from_hsl(hue, 0.7, 0.5).to_hex()


class MagnitudeWidth(WidthStrategy):
    def __init__(self, min_width: float = 0.5, max_width: float = 3.0, mag_range: tuple[float, float] = (0.0, 1.0)):
        self.min_width = min_width
        self.max_width = max_width
        self.mag_range = mag_range

    def get_width(self, streamline: np.ndarray, index: int, magnitudes: np.ndarray | None = None) -> float:
        if magnitudes is None:
            return self.min_width
        mag = magnitudes[index]
        # Normalize magnitude
        t = (mag - self.mag_range[0]) / (self.mag_range[1] - self.mag_range[0])
        t = np.clip(t, 0.0, 1.0)
        return self.min_width + t * (self.max_width - self.min_width)


class MagnitudeColor(ColorStrategy):
    def __init__(self, color1: str = "#0000ff", color2: str = "#ff0000", mag_range: tuple[float, float] = (0.0, 1.0)):
        self.color1 = color1
        self.color2 = color2
        self.mag_range = mag_range

    def get_color(self, streamline: np.ndarray, index: int, magnitudes: np.ndarray | None = None) -> str:
        if magnitudes is None:
            return self.color1
        mag = magnitudes[index]
        t = (mag - self.mag_range[0]) / (self.mag_range[1] - self.mag_range[0])
        t = np.clip(t, 0.0, 1.0)

        from bp_designs.core.color import Color

        c1 = Color.from_hex(self.color1)
        c2 = Color.from_hex(self.color2)
        return Color.lerp(c1, c2, t).to_hex()


class FlowStyle(RenderStyle):
    """Structured rendering parameters for flow fields."""

    model_config = {"arbitrary_types_allowed": True}

    min_thickness: float = 0.5
    max_thickness: float = 2.0
    taper: bool = True
    color_mode: str = "constant"  # "constant" or "angle"
    color: str = "#000000"
    epsilon: float = 0.1  # RDP simplification threshold

    # New strategy-based fields
    width_strategy: WidthStrategy | None = None
    color_strategy: ColorStrategy | None = None


@dataclass
class StreamlinePattern(Pattern):
    """A pattern consisting of multiple streamlines (polylines)."""

    streamlines: list[np.ndarray] = field(default_factory=list)
    magnitudes: list[np.ndarray] = field(default_factory=list)
    # Metadata for rendering
    widths: list[float] | None = None
    colors: list[str] | None = None

    def to_geometry(self, canvas: Canvas | None = None, epsilon: float | None = None) -> Geometry:
        """Convert streamlines to a Polyline geometry.
        
        Args:
            canvas: Optional canvas for coordinate resolution.
            epsilon: Simplification threshold (RDP). 0.0 means no simplification.
        """
        if epsilon is None:
            epsilon = self.render_params.get("epsilon", 0.0)

        lines = self.streamlines
        if epsilon > 0:
            lines = [self._rdp(line, epsilon) for line in lines]
        return Polyline(polylines=lines)

    def _rdp(self, points: np.ndarray, epsilon: float) -> np.ndarray:
        """Ramer-Douglas-Peucker algorithm for polyline simplification."""
        if len(points) < 3:
            return points

        # Find the point with the maximum distance
        dmax = 0
        index = 0
        end = len(points) - 1
        for i in range(1, end):
            d = self._perpendicular_distance(points[i], points[0], points[end])
            if d > dmax:
                index = i
                dmax = d

        # If max distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            # Recursive call
            rec_results1 = self._rdp(points[:index+1], epsilon)
            rec_results2 = self._rdp(points[index:], epsilon)

            # Build the result list
            return np.vstack((rec_results1[:-1], rec_results2))
        else:
            return np.vstack((points[0], points[end]))

    def _perpendicular_distance(self, p, a, b):
        """Calculate perpendicular distance from point p to line segment ab."""
        if np.all(a == b):
            return np.linalg.norm(p - a)

        return np.abs(np.cross(b - a, a - p)) / np.linalg.norm(b - a)

    def render(self, context: RenderingContext, style: FlowStyle | None = None, **kwargs):
        """Render streamlines into the provided context."""
        params = self.render_params.copy()
        params.update(kwargs)

        if style is None:
            style = FlowStyle.from_dict(params)

        # Initialize strategies if not provided
        width_strategy = style.width_strategy
        if width_strategy is None:
            if style.taper:
                width_strategy = TaperedWidth(style.min_thickness, style.max_thickness)
            else:
                width_strategy = ConstantWidth(style.max_thickness)

        color_strategy = style.color_strategy
        if color_strategy is None:
            if style.color_mode == "angle":
                color_strategy = AngleColor()
            else:
                color_strategy = ConstantColor(style.color)

        # Filter kwargs to only include valid SVG attributes
        known_style_fields = set(FlowStyle.model_fields.keys())
        svg_kwargs = {k: v for k, v in kwargs.items() if k not in known_style_fields and k != "lighting"}

        # We use the original streamlines for rendering if we need magnitudes,
        # because RDP simplification might not align with magnitude indices.
        # For now, let's assume we use original streamlines if magnitude mapping is used.
        # If epsilon is 0, to_geometry returns original streamlines.
        geom = self.to_geometry(epsilon=style.epsilon)

        for s_idx, streamline in enumerate(geom.polylines):
            if len(streamline) < 2:
                continue

            # Get magnitudes for this streamline if available
            mags = self.magnitudes[s_idx] if s_idx < len(self.magnitudes) else None

            # Render segment by segment
            for i in range(len(streamline) - 1):
                p1, p2 = streamline[i], streamline[i + 1]

                width = width_strategy.get_width(streamline, i, mags)
                color = color_strategy.get_color(streamline, i, mags)

                context.add(
                    context.dwg.polyline(
                        points=[(float(p[0]), float(p[1])) for p in [p1, p2]],
                        stroke=color,
                        stroke_width=width,
                        fill="none",
                        **svg_kwargs,
                    )
                )

    def __str__(self) -> str:
        return f"StreamlinePattern(n={len(self.streamlines)})"
