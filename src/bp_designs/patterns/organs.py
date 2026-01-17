"""Organ patterns and generators for organic details.

Implements the Pattern-Generator architecture for organs (leaves, blossoms, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import svgwrite

from bp_designs.core.generator import Generator
from bp_designs.core.geometry import Geometry, Polygon
from bp_designs.core.pattern import Pattern

if TYPE_CHECKING:
    from bp_designs.core.color import Color
    from bp_designs.core.renderer import RenderingContext, RenderStyle


class OrganPattern(Pattern):
    """Semantic representation of an organic detail.

    An OrganPattern defines the 'blueprint' of an organ, which can then be
    instantiated and transformed by a host pattern (like a BranchNetwork).
    """

    def __init__(self, name: str, base_color: str | Color, scale: float = 1.0):
        self.name = name
        self.base_color = base_color
        self.scale = scale

    @abstractmethod
    def render(
        self,
        context: RenderingContext,
        position: np.ndarray,
        style: RenderStyle | None = None,
        color: str | None = None,
        orientation: float = 0.0,
        scale_factor: float = 1.0,
        **kwargs,
    ):
        """Render the organ to a rendering context.

        Args:
            context: The rendering context to add to.
            position: (2,) array for the attachment point.
            style: Structured rendering parameters.
            color: Override color (if None, uses base_color).
            orientation: Rotation in degrees.
            scale_factor: Additional scaling (e.g., based on branch thickness).
            **kwargs: Additional rendering parameters.
        """
        pass

    def to_geometry(self, canvas=None) -> Geometry:
        """Convert to geometric representation (default is a simple point)."""
        # Most organs are small enough to be treated as points in global geometry
        # unless we specifically need their outlines.
        return Polygon(coords=np.array([[0.0, 0.0]]))

    def to_svg(self, **kwargs) -> str:
        """Convert to a standalone SVG string (for preview)."""
        from bp_designs.core.renderer import RenderingContext

        dwg = svgwrite.Drawing(size=("100px", "100px"), viewBox="-50 -50 100 100")
        context = RenderingContext(dwg)
        self.render(context, np.array([0.0, 0.0]), **kwargs)
        return dwg.tostring()

    def __str__(self) -> str:
        return f"OrganPattern({self.name}, scale={self.scale})"


class OrganGenerator(Generator, ABC):
    """Base class for algorithms that generate organ patterns."""

    @abstractmethod
    def generate_pattern(self, **kwargs) -> OrganPattern:
        """Generate an organ pattern."""
        pass


class CircleOrganPattern(OrganPattern):
    """Simple circular organ (e.g., fruit or berry)."""

    def __init__(self, name: str, base_color: str | Color, scale: float = 1.0, radius: float = 2.0):
        super().__init__(name=name, base_color=base_color, scale=scale)
        self.radius = radius

    def render(
        self,
        context,
        position,
        style=None,
        color=None,
        orientation=0.0,
        scale_factor=1.0,
        **kwargs,
    ):
        fill_color = str(color) if color is not None else str(self.base_color)
        svg_attrs = style.get_svg_attributes() if style else {}
        context.add(
            context.dwg.circle(
                center=(float(position[0]), float(position[1])),
                r=self.radius * self.scale * scale_factor,
                fill=fill_color,
                **svg_attrs,
            )
        )


class LeafOrganPattern(OrganPattern):
    """Simple leaf-shaped organ."""

    def __init__(
        self,
        name: str,
        base_color: str | Color,
        scale: float = 1.0,
        angle_offset: float = 0.0,
        jitter: float = 0.0,
    ):
        super().__init__(name=name, base_color=base_color, scale=scale)
        self.angle_offset = angle_offset
        self.jitter = jitter

    def render(
        self,
        context,
        position,
        style=None,
        color=None,
        orientation=0.0,
        scale_factor=1.0,
        **kwargs,
    ):
        fill_color = str(color) if color is not None else str(self.base_color)
        svg_attrs = style.get_svg_attributes() if style else {}

        # Add some random jitter to orientation if requested
        final_angle = orientation + self.angle_offset
        if self.jitter > 0:
            rng = np.random.default_rng(int(position[0] * 1000 + position[1]))
            final_angle += rng.uniform(-self.jitter, self.jitter)

        # Simple diamond/leaf shape
        s = self.scale * scale_factor
        # Ensure minimum visibility
        s = max(s, 1.0)
        points = [(0, 0), (s, s / 3), (s * 1.5, 0), (s, -s / 3)]

        # Rotate and translate points
        import math

        rad = math.radians(final_angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        transformed_points = []
        for px, py in points:
            tx = px * cos_a - py * sin_a + position[0]
            ty = px * sin_a + py * cos_a + position[1]
            transformed_points.append((float(tx), float(ty)))

        context.add(context.dwg.polygon(points=transformed_points, fill=fill_color, **svg_attrs))


class LeafGenerator(OrganGenerator):
    """Generator for leaf patterns."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate_pattern(
        self,
        base_color: str | Color = "#2d5a27",
        scale: float = 5.0,
        angle_offset: float = 0.0,
        jitter: float = 0.0,
        **kwargs,
    ) -> LeafOrganPattern:
        return LeafOrganPattern(
            name="Simple Leaf",
            base_color=base_color,
            scale=scale,
            angle_offset=angle_offset,
            jitter=jitter,
        )
