from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from bp_designs.core.geometry import Canvas, Geometry, Polyline
from bp_designs.core.pattern import Pattern
from bp_designs.core.renderer import RenderStyle

if TYPE_CHECKING:
    from bp_designs.core.renderer import RenderingContext


class FlowStyle(RenderStyle):
    """Structured rendering parameters for flow fields."""
    min_thickness: float = 0.5
    max_thickness: float = 2.0
    taper: bool = True
    color_mode: str = "constant"  # "constant" or "angle"
    color: str = "#000000"
    epsilon: float = 0.1  # RDP simplification threshold

@dataclass
class StreamlinePattern(Pattern):
    """A pattern consisting of multiple streamlines (polylines)."""
    streamlines: list[np.ndarray] = field(default_factory=list)
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

        # Filter kwargs to only include valid SVG attributes
        # We use the get_svg_attributes helper from RenderStyle if possible,
        # but here we want to filter the kwargs passed to render.
        known_style_fields = set(FlowStyle.model_fields.keys())
        svg_kwargs = {k: v for k, v in kwargs.items() if k not in known_style_fields and k != "lighting"}

        geom = self.to_geometry(epsilon=style.epsilon)

        for streamline in geom.polylines:
            if len(streamline) < 2:
                continue

            # Calculate colors and widths per segment if needed
            if style.taper or style.color_mode == "angle":
                # Render segment by segment for varying width/color
                for i in range(len(streamline) - 1):
                    p1, p2 = streamline[i], streamline[i+1]

                    # Width
                    if style.taper:
                        # Taper from max to min along the streamline
                        progress = i / (len(streamline) - 1)
                        width = style.max_thickness * (1.0 - progress) + style.min_thickness * progress
                    else:
                        width = style.max_thickness

                    # Color
                    if style.color_mode == "angle":
                        v = p2 - p1
                        angle = np.arctan2(v[1], v[0])
                        # Map angle [-pi, pi] to [0, 1] for color mapping
                        hue = (angle + np.pi) / (2 * np.pi)
                        from bp_designs.core.color import Color
                        color = Color.from_hsl(hue, 0.7, 0.5).to_hex()
                    else:
                        color = style.color

                    context.add(context.dwg.polyline(
                        points=[(float(p[0]), float(p[1])) for p in [p1, p2]],
                        stroke=color,
                        stroke_width=width,
                        fill="none",
                        **svg_kwargs
                    ))
            else:
                # Constant width and color, can draw whole polyline at once
                context.add(context.dwg.polyline(
                    points=[(float(p[0]), float(p[1])) for p in streamline],
                    stroke=style.color,
                    stroke_width=style.max_thickness,
                    fill="none",
                    **svg_kwargs
                ))

    def __str__(self) -> str:
        return f"StreamlinePattern(n={len(self.streamlines)})"
