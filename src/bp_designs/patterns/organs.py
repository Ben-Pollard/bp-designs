"""Organ patterns and generators for organic details.

Implements the Pattern-Generator architecture for organs (leaves, blossoms, etc.).
"""

from __future__ import annotations

import math
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
        if context.lighting:
            fill_color = context.lighting.get_fill(
                fill_color, {"type": "organ", "position": position}
            )
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
        if context.lighting:
            fill_color = context.lighting.get_fill(
                fill_color, {"type": "organ", "position": position, "angle": orientation}
            )
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


class BlossomOrganPattern(OrganPattern):
    """Procedural blossom with multiple rings of petals.

    Implements a radial layout of petals with support for multiple rings,
    rotation offsets between rings, and organic petal shapes.
    """

    def __init__(
        self,
        name: str,
        base_color: str | Color,
        scale: float = 1.0,
        num_rings: int = 2,
        base_petal_count: int | list[int] = 5,
        petal_width: float = 3.0,
        petal_height: float = 6.0,
        inner_radius: float = 0.0,
        ring_spacing: float = 2.0,
        rotation_offset: float = 36.0,  # Degrees
        petal_shape: str = "teardrop",
        center_radius: float = 1.5,
        center_color: str | Color | None = None,
        jitter: float = 0.1,
        overlap: float = 1.2,
    ):
        super().__init__(name=name, base_color=base_color, scale=scale)
        self.num_rings = num_rings
        self.base_petal_count = base_petal_count
        self.petal_width = petal_width
        self.petal_height = petal_height
        self.inner_radius = inner_radius
        self.ring_spacing = ring_spacing
        self.rotation_offset = rotation_offset
        self.petal_shape = petal_shape
        self.center_radius = center_radius
        self.center_color = center_color
        self.jitter = jitter
        self.overlap = overlap

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
        s = self.scale * scale_factor

        # Create a group for the blossom to handle global rotation and position
        group = context.push_group(
            f"blossom_{id(self)}",
            transform=f"translate({position[0]},{position[1]}) rotate({orientation})",
        )

        rng = np.random.default_rng(int(position[0] * 1000 + position[1]))

        # Render rings from outside in for correct overlapping
        for r in range(self.num_rings - 1, -1, -1):
            ring_radius = (self.inner_radius + r * self.ring_spacing) * s
            ring_rotation = r * self.rotation_offset

            # Determine petal count for this ring
            if isinstance(self.base_petal_count, list):
                if r < len(self.base_petal_count):
                    curr_petals = self.base_petal_count[r]
                else:
                    curr_petals = self.base_petal_count[-1]
            else:
                # Calculate petals needed to cover the circumference at this radius
                # Circumference = 2 * pi * radius
                # Petal width = self.petal_width * s
                # We want enough petals to cover the circumference with some overlap
                petal_w = self.petal_width * s
                circumference = 2 * math.pi * ring_radius

                # Number of petals to cover circumference once: circumference / petal_w
                # Apply overlap factor
                curr_petals = max(
                    self.base_petal_count,
                    int(math.ceil((circumference / petal_w) * self.overlap))
                )

            for p in range(curr_petals):
                angle = (360.0 / curr_petals) * p + ring_rotation
                # Add jitter
                if self.jitter > 0:
                    angle += rng.uniform(-5, 5) * self.jitter
                    curr_radius = ring_radius * (1 + rng.uniform(-0.1, 0.1) * self.jitter)
                else:
                    curr_radius = ring_radius

                rad = math.radians(angle)
                px = curr_radius * math.cos(rad)
                py = curr_radius * math.sin(rad)

                # Petal orientation: pointing away from center
                petal_angle = angle + 90

                self._render_petal(
                    context,
                    (px, py),
                    petal_angle,
                    s,
                    fill_color,
                    svg_attrs,
                    rng if self.jitter > 0 else None,
                )

        # Render center
        c_color = str(self.center_color) if self.center_color else "#ffcc00"
        if context.lighting:
            c_color = context.lighting.get_fill(
                c_color, {"type": "organ", "position": position, "part": "center"}
            )

        context.add(
            context.dwg.circle(
                center=(0, 0),
                r=self.center_radius * s,
                fill=c_color,
                **svg_attrs,
            )
        )

        context.pop_group()

    def _render_petal(self, context, pos, angle, scale, color, attrs, rng):
        """Render a single petal."""
        w = self.petal_width * scale
        h = self.petal_height * scale

        if rng:
            w *= 1 + rng.uniform(-0.1, 0.1) * self.jitter
            h *= 1 + rng.uniform(-0.1, 0.1) * self.jitter

        # Use a group for the petal to handle its local transform
        # We rotate it so it points outwards
        p_group = context.push_group(
            "petal",
            transform=f"translate({pos[0]},{pos[1]}) rotate({angle})",
        )

        fill_color = color
        if context.lighting:
            # Pass some info to lighting to allow per-petal variation if supported
            fill_color = context.lighting.get_fill(
                color, {"type": "organ", "part": "petal", "angle": angle}
            )

        if self.petal_shape == "teardrop":
            # Teardrop path: starts at base, bulges, ends at tip
            path_data = f"M 0,0 C {w},{-h/3} {w/2},{-h} 0,{-h} C {-w/2},{-h} {-w},{-h/3} 0,0 Z"
            context.add(context.dwg.path(d=path_data, fill=fill_color, **attrs))
        elif self.petal_shape == "heart":
            # Heart-ish shape
            path_data = (
                f"M 0,0 C {w},{-h/4} {w},{-h} 0,{-h*0.8} C {-w},{-h} {-w},{-h/4} 0,0 Z"
            )
            context.add(context.dwg.path(d=path_data, fill=fill_color, **attrs))
        else:
            # Default to oval
            context.add(
                context.dwg.ellipse(
                    center=(0, -h / 2),
                    r=(w / 2, h / 2),
                    fill=fill_color,
                    **attrs,
                )
            )

        context.pop_group()


class BlossomGenerator(OrganGenerator):
    """Generator for blossom patterns."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate_pattern(
        self,
        base_color: str | Color = "#ff69b4",
        scale: float = 1.0,
        num_rings: int = 2,
        base_petal_count: int | list[int] = 5,
        petal_width: float = 3.0,
        petal_height: float = 6.0,
        inner_radius: float = 0.0,
        ring_spacing: float = 2.0,
        rotation_offset: float = 36.0,
        petal_shape: str = "teardrop",
        center_radius: float = 1.5,
        center_color: str | Color | None = None,
        jitter: float = 0.1,
        overlap: float = 1.2,
        **kwargs,
    ) -> BlossomOrganPattern:
        return BlossomOrganPattern(
            name="Procedural Blossom",
            base_color=base_color,
            scale=scale,
            num_rings=num_rings,
            base_petal_count=base_petal_count,
            petal_width=petal_width,
            petal_height=petal_height,
            inner_radius=inner_radius,
            ring_spacing=ring_spacing,
            rotation_offset=rotation_offset,
            petal_shape=petal_shape,
            center_radius=center_radius,
            center_color=center_color,
            jitter=jitter,
            overlap=overlap,
        )
