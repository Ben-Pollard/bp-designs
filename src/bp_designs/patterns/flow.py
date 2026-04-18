from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from bp_designs.core.geometry import Canvas, Geometry, Polyline
from bp_designs.core.pattern import Pattern
from bp_designs.core.renderer import RenderStyle

if TYPE_CHECKING:
    from bp_designs.core.color import Palette
    from bp_designs.core.renderer import RenderingContext


class WidthStrategy(ABC):
    """Abstract base class for streamline width strategies."""

    def __init__(self, n_bins: int | None = None):
        self.n_bins = n_bins

    @abstractmethod
    def get_width(self, streamline: np.ndarray, index: int, magnitudes: np.ndarray | None = None) -> float:
        pass

    def get_widths(self, streamline: np.ndarray, magnitudes: np.ndarray | None = None) -> np.ndarray:
        """Vectorized width calculation for the whole streamline."""
        return np.array(
            [self.get_width(streamline, i, magnitudes) for i in range(len(streamline) - 1)],
            dtype=float,
        )


class ConstantWidth(WidthStrategy):
    def __init__(self, width: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.width = width

    def get_width(self, streamline: np.ndarray, index: int, magnitudes: np.ndarray | None = None) -> float:
        return self.width

    def get_widths(self, streamline: np.ndarray, magnitudes: np.ndarray | None = None) -> np.ndarray:
        return np.full(len(streamline) - 1, self.width, dtype=float)


class TaperedWidth(WidthStrategy):
    def __init__(self, min_width: float = 0.5, max_width: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.min_width = min_width
        self.max_width = max_width

    def get_width(self, streamline: np.ndarray, index: int, magnitudes: np.ndarray | None = None) -> float:
        if len(streamline) < 2:
            return self.max_width
        progress = index / (len(streamline) - 1)

        if self.n_bins:
            progress = np.round(progress * (self.n_bins - 1)) / (self.n_bins - 1)

        return self.max_width * (1.0 - progress) + self.min_width * progress

    def get_widths(self, streamline: np.ndarray, magnitudes: np.ndarray | None = None) -> np.ndarray:
        n = len(streamline)
        if n < 2:
            return np.array([], dtype=float)
        t = np.linspace(0, 1, n - 1)

        if self.n_bins:
            t = np.round(t * (self.n_bins - 1)) / (self.n_bins - 1)

        return self.max_width * (1.0 - t) + self.min_width * t


class ColorStrategy(ABC):
    """Abstract base class for streamline color strategies."""

    def __init__(self, n_bins: int | None = None):
        self.n_bins = n_bins

    @abstractmethod
    def get_color(self, streamline: np.ndarray, index: int, magnitudes: np.ndarray | None = None) -> str:
        pass

    def get_colors(self, streamline: np.ndarray, magnitudes: np.ndarray | None = None) -> list[str]:
        """Vectorized color calculation for the whole streamline."""
        return [self.get_color(streamline, i, magnitudes) for i in range(len(streamline) - 1)]


class ConstantColor(ColorStrategy):
    def __init__(self, color: str = "#000000", **kwargs):
        super().__init__(**kwargs)
        self.color = color

    def get_color(self, streamline: np.ndarray, index: int, magnitudes: np.ndarray | None = None) -> str:
        return self.color

    def get_colors(self, streamline: np.ndarray, magnitudes: np.ndarray | None = None) -> list[str]:
        return [self.color] * (len(streamline) - 1)


class AngleColor(ColorStrategy):
    def get_color(self, streamline: np.ndarray, index: int, magnitudes: np.ndarray | None = None) -> str:
        p1, p2 = streamline[index], streamline[index + 1]
        v = p2 - p1
        angle = np.arctan2(v[1], v[0])
        t = (angle + np.pi) / (2 * np.pi)

        if self.n_bins:
            t = np.round(t * (self.n_bins - 1)) / (self.n_bins - 1)

        from bp_designs.core.color import Color

        return Color.from_hsl(t, 0.7, 0.5).to_hex()

    def get_colors(self, streamline: np.ndarray, magnitudes: np.ndarray | None = None) -> list[str]:
        if len(streamline) < 2:
            return []
        diffs = np.diff(streamline, axis=0)
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        t_vals = (angles + np.pi) / (2 * np.pi)

        if self.n_bins:
            t_vals = np.round(t_vals * (self.n_bins - 1)) / (self.n_bins - 1)

        from bp_designs.core.color import Color

        return [Color.from_hsl(t, 0.7, 0.5).to_hex() for t in t_vals]


class MagnitudeWidth(WidthStrategy):
    def __init__(
        self,
        min_width: float = 0.5,
        max_width: float = 3.0,
        mag_range: tuple[float, float] = (0.0, 1.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
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

        if self.n_bins:
            t = np.round(t * (self.n_bins - 1)) / (self.n_bins - 1)

        return self.min_width + t * (self.max_width - self.min_width)

    def get_widths(self, streamline: np.ndarray, magnitudes: np.ndarray | None = None) -> np.ndarray:
        if magnitudes is None or len(magnitudes) < 1:
            return np.full(len(streamline) - 1, self.min_width)
        t = (magnitudes[: len(streamline) - 1] - self.mag_range[0]) / (self.mag_range[1] - self.mag_range[0])
        t = np.clip(t, 0.0, 1.0)

        if self.n_bins:
            t = np.round(t * (self.n_bins - 1)) / (self.n_bins - 1)

        return self.min_width + t * (self.max_width - self.min_width)


class MagnitudeColor(ColorStrategy):
    def __init__(
        self,
        color1: str = "#0000ff",
        color2: str = "#ff0000",
        mag_range: tuple[float, float] = (0.0, 1.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.color1 = color1
        self.color2 = color2
        self.mag_range = mag_range

    def get_color(self, streamline: np.ndarray, index: int, magnitudes: np.ndarray | None = None) -> str:
        if magnitudes is None:
            return self.color1
        mag = magnitudes[index]
        t = (mag - self.mag_range[0]) / (self.mag_range[1] - self.mag_range[0])
        t = np.clip(t, 0.0, 1.0)

        if self.n_bins:
            t = np.round(t * (self.n_bins - 1)) / (self.n_bins - 1)

        from bp_designs.core.color import Color

        c1 = Color.from_hex(self.color1)
        c2 = Color.from_hex(self.color2)
        return Color.lerp(c1, c2, t).to_hex()

    def get_colors(self, streamline: np.ndarray, magnitudes: np.ndarray | None = None) -> list[str]:
        if magnitudes is None or len(magnitudes) < 1:
            return [self.color1] * (len(streamline) - 1)

        t_vals = (magnitudes[: len(streamline) - 1] - self.mag_range[0]) / (
            self.mag_range[1] - self.mag_range[0]
        )
        t_vals = np.clip(t_vals, 0.0, 1.0)

        if self.n_bins:
            t_vals = np.round(t_vals * (self.n_bins - 1)) / (self.n_bins - 1)

        from bp_designs.core.color import Color

        c1 = Color.from_hex(self.color1)
        c2 = Color.from_hex(self.color2)
        return [Color.lerp(c1, c2, t).to_hex() for t in t_vals]


class PaletteMapColor(ColorStrategy):
    """Maps a field property to a color from a palette."""

    def __init__(
        self,
        palette: Palette | str,
        property: str = "angle",
        range: tuple[float, float] = (0.0, 1.0),
        interpolate: bool = True,
        per_segment: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(palette, str):
            from bp_designs.core.color import MasterPalettes

            self.palette = MasterPalettes.get(palette)
        else:
            self.palette = palette
        self.property = property
        self.range = range
        self.interpolate = interpolate
        self.per_segment = per_segment

    def get_color(self, streamline: np.ndarray, index: int, magnitudes: np.ndarray | None = None) -> str:
        # If not per_segment, always use the first point/magnitude
        eval_idx = index if self.per_segment else 0

        if self.property == "angle":
            # Angle is tricky for per_streamline: we use the first segment's angle
            p1, p2 = streamline[eval_idx], streamline[eval_idx + 1]
            v = p2 - p1
            angle = np.arctan2(v[1], v[0])
            t = (angle + np.pi) / (2 * np.pi)
        elif self.property == "magnitude":
            if magnitudes is None:
                t = 0.5
            else:
                mag = magnitudes[eval_idx]
                t = (mag - self.range[0]) / (self.range[1] - self.range[0])
        elif self.property == "position_x":
            t = (streamline[eval_idx, 0] - self.range[0]) / (self.range[1] - self.range[0])
        elif self.property == "position_y":
            t = (streamline[eval_idx, 1] - self.range[0]) / (self.range[1] - self.range[0])
        elif self.property == "start_x":
            t = (streamline[0, 0] - self.range[0]) / (self.range[1] - self.range[0])
        elif self.property == "start_y":
            t = (streamline[0, 1] - self.range[0]) / (self.range[1] - self.range[0])
        elif self.property == "start_magnitude":
            if magnitudes is None:
                t = 0.5
            else:
                mag = magnitudes[0]
                t = (mag - self.range[0]) / (self.range[1] - self.range[0])
        else:
            t = 0.5

        t = np.clip(t, 0.0, 1.0)

        if self.n_bins:
            t = np.round(t * (self.n_bins - 1)) / (self.n_bins - 1)

        if self.interpolate:
            return self.palette.lerp_at(t).to_hex()
        else:
            return self.palette.get_at(t).to_hex()

    def get_colors(self, streamline: np.ndarray, magnitudes: np.ndarray | None = None) -> list[str]:
        n = len(streamline)
        if n < 2:
            return []

        # Optimization: if not per_segment, calculate once and repeat
        if not self.per_segment:
            color = self.get_color(streamline, 0, magnitudes)
            return [color] * (n - 1)

        # 1. Calculate normalized parameter t for the whole streamline
        if self.property == "angle":
            diffs = np.diff(streamline, axis=0)
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            t_vals = (angles + np.pi) / (2 * np.pi)
        elif self.property == "magnitude":
            if magnitudes is None:
                t_vals = np.full(n - 1, 0.5)
            else:
                t_vals = (magnitudes[: n - 1] - self.range[0]) / (self.range[1] - self.range[0])
        elif self.property == "position_x":
            t_vals = (streamline[: n - 1, 0] - self.range[0]) / (self.range[1] - self.range[0])
        elif self.property == "position_y":
            t_vals = (streamline[: n - 1, 1] - self.range[0]) / (self.range[1] - self.range[0])
        elif self.property == "start_x":
            t_vals = np.full(n - 1, (streamline[0, 0] - self.range[0]) / (self.range[1] - self.range[0]))
        elif self.property == "start_y":
            t_vals = np.full(n - 1, (streamline[0, 1] - self.range[0]) / (self.range[1] - self.range[0]))
        elif self.property == "start_magnitude":
            if magnitudes is None:
                t_vals = np.full(n - 1, 0.5)
            else:
                t_vals = np.full(n - 1, (magnitudes[0] - self.range[0]) / (self.range[1] - self.range[0]))
        else:
            t_vals = np.full(n - 1, 0.5)

        t_vals = np.clip(t_vals, 0.0, 1.0)

        if self.n_bins:
            t_vals = np.round(t_vals * (self.n_bins - 1)) / (self.n_bins - 1)

        # 2. Map t values to hex colors
        if self.interpolate:
            return [self.palette.lerp_at(t).to_hex() for t in t_vals]
        else:
            return [self.palette.get_at(t).to_hex() for t in t_vals]


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
            rec_results1 = self._rdp(points[: index + 1], epsilon)
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
        import time

        start_render = time.time()
        params = self.render_params.copy()
        params.update(kwargs)

        if style is None:
            style = FlowStyle.from_dict(params)

        # Initialize strategies if not provided
        width_strategy = style.width_strategy
        if width_strategy is None:
            if style.taper:
                width_strategy = TaperedWidth(min_width=style.min_thickness, max_width=style.max_thickness)
            else:
                width_strategy = ConstantWidth(width=style.max_thickness)

        color_strategy = style.color_strategy
        if color_strategy is None:
            if style.color_mode == "angle":
                color_strategy = AngleColor()
            else:
                color_strategy = ConstantColor(color=style.color)

        # Filter kwargs to only include valid SVG attributes
        known_style_fields = set(FlowStyle.model_fields.keys())
        svg_kwargs = {k: v for k, v in kwargs.items() if k not in known_style_fields and k != "lighting"}

        # Get geometry (optionally simplified)
        geom = self.to_geometry(epsilon=style.epsilon)

        total_segments = 0
        total_objects = 0

        for s_idx, streamline in enumerate(geom.polylines):
            if len(streamline) < 2:
                continue

            # Get magnitudes for this streamline if available
            mags = self.magnitudes[s_idx] if s_idx < len(self.magnitudes) else None

            # Pre-calculate all widths and colors for the streamline
            # Round widths to 2 decimal places to facilitate batching
            widths = np.round(width_strategy.get_widths(streamline, mags), 2)
            colors = color_strategy.get_colors(streamline, mags)

            # Optimization: If the whole streamline has the same color and width,
            # we can draw it as a single polyline.
            is_uniform = False
            if len(widths) > 0:
                first_w = widths[0]
                first_c = colors[0]
                if np.all(widths == first_w) and all(c == first_c for c in colors):
                    is_uniform = True

            if is_uniform:
                context.add(
                    context.dwg.polyline(
                        points=[(round(float(p[0]), 2), round(float(p[1]), 2)) for p in streamline],
                        stroke=colors[0],
                        stroke_width=float(widths[0]),
                        fill="none",
                        **svg_kwargs,
                    )
                )
                total_segments += len(streamline) - 1
                total_objects += 1
                continue

            # Render by grouping consecutive segments with same color/width
            start_idx = 0
            while start_idx < len(streamline) - 1:
                current_color = colors[start_idx]
                current_width = widths[start_idx]

                # Find how many segments share this style
                end_idx = start_idx + 1
                while (
                    end_idx < len(streamline) - 1
                    and colors[end_idx] == current_color
                    and widths[end_idx] == current_width
                ):
                    end_idx += 1

                # Add as a single polyline
                pts = streamline[start_idx : end_idx + 1]
                context.add(
                    context.dwg.polyline(
                        points=[(round(float(p[0]), 2), round(float(p[1]), 2)) for p in pts],
                        stroke=current_color,
                        stroke_width=float(current_width),
                        fill="none",
                        **svg_kwargs,
                    )
                )
                total_segments += end_idx - start_idx
                total_objects += 1
                start_idx = end_idx

        end_render = time.time()
        print(
            f"Rendered {total_segments} segments using {total_objects} SVG objects in {end_render - start_render:.2f}s"
        )

    def __str__(self) -> str:
        return f"StreamlinePattern(n={len(self.streamlines)})"
