"""Lighting models for artistic shading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from bp_designs.core.color import Color
    from bp_designs.core.renderer import RenderingContext


class LightingModel(ABC):
    """Base class for lighting models that provide shading information."""

    @abstractmethod
    def setup(self, context: RenderingContext):
        """Initialize lighting in the rendering context (e.g., add SVG defs)."""
        pass

    @abstractmethod
    def get_fill(self, base_color: Color | str, geometry_info: dict[str, Any]) -> str:
        """Return an SVG fill value (color or url(#id))."""
        pass


class DirectionalLighting(LightingModel):
    """Simple directional lighting model for volume-based shading."""

    def __init__(
        self,
        light_direction: np.ndarray = np.array([-1.0, -1.0]),
        highlight_amount: float = 0.2,
        shadow_amount: float = 0.2,
    ):
        """Initialize directional lighting.

        Args:
            light_direction: 2D vector pointing toward the light source.
            highlight_amount: Lightness increase for highlights.
            shadow_amount: Lightness decrease for shadows.
        """
        # Normalize light direction
        norm = np.linalg.norm(light_direction)
        self.light_direction = light_direction / norm if norm > 0 else np.array([-1.0, -1.0])
        self.highlight_amount = highlight_amount
        self.shadow_amount = shadow_amount
        self._gradient_cache: dict[str, str] = {}

    def setup(self, context: RenderingContext):
        """Initialize lighting in the rendering context."""
        self._context = context

    def get_fill(self, base_color: Any, geometry_info: dict[str, Any]) -> str:
        """Get fill for a specific geometry.

        Supported geometry_info keys:
        - 'type': 'branch', 'organ', 'background'
        - 'vector': (2,) segment vector (for branches)
        - 'position': (2,) position (for organs)
        """
        from bp_designs.core.color import Color

        # Convert base_color to Color object if it's a string
        if isinstance(base_color, str):
            try:
                color_obj = Color.from_hex(base_color)
            except (ValueError, IndexError):
                # Fallback if not a valid hex string
                return base_color
        else:
            color_obj = base_color

        g_type = geometry_info.get("type", "background")

        if g_type == "branch" and "vector" in geometry_info:
            return self._get_branch_fill(color_obj, geometry_info["vector"])
        elif g_type == "organ":
            return self._get_organ_fill(color_obj, geometry_info)
        elif g_type == "background":
            return self._get_background_fill(color_obj)
        elif g_type == "global":
            return self._get_global_fill(color_obj)
        else:
            return str(color_obj)

    def _get_branch_fill(self, color: Color, segment_v: np.ndarray) -> str:
        """Create a cylindrical gradient perpendicular to the branch segment."""
        if not hasattr(self, "_context"):
            return str(color)

        # Calculate perpendicular vector for the gradient direction
        length = np.linalg.norm(segment_v)
        if length == 0:
            return str(color)

        # Perpendicular vector (normal)
        normal = np.array([-segment_v[1], segment_v[0]]) / length

        # Dot product with light direction to find highlight/shadow bias
        bias = np.dot(normal, self.light_direction)

        # Create a unique ID for this gradient based on color and bias
        # Use 'n' for negative sign in ID
        bias_str = f"{bias:.2f}".replace("-", "n")
        grad_id = f"branch_grad_{color.to_hex().replace('#', '')}_{bias_str}"

        if grad_id not in self._gradient_cache:
            # Create highlight and shadow colors
            highlight = color.lighten(self.highlight_amount)
            shadow = color.darken(self.shadow_amount)

            # Define gradient direction based on normal and bias
            # We want the highlight on the side facing the light
            x1, y1 = 0.5 - normal[0] * 0.5, 0.5 - normal[1] * 0.5
            x2, y2 = 0.5 + normal[0] * 0.5, 0.5 + normal[1] * 0.5

            grad = self._context.dwg.linearGradient(
                id=grad_id,
                start=(f"{x1*100:.1f}%", f"{y1*100:.1f}%"),
                end=(f"{x2*100:.1f}%", f"{y2*100:.1f}%"),
                gradientUnits="objectBoundingBox",
            )

            # Add stops: Shadow -> Base -> Highlight (or vice versa based on bias)
            if bias > 0:
                grad.add_stop_color(offset="0%", color=str(highlight))
                grad.add_stop_color(offset="50%", color=str(color))
                grad.add_stop_color(offset="100%", color=str(shadow))
            else:
                grad.add_stop_color(offset="0%", color=str(shadow))
                grad.add_stop_color(offset="50%", color=str(color))
                grad.add_stop_color(offset="100%", color=str(highlight))

            self._context.dwg.defs.add(grad)
            self._gradient_cache[grad_id] = f"url(#{grad_id})"

        return self._gradient_cache[grad_id]

    def _get_background_fill(self, color: Color) -> str:
        """Create a global atmospheric gradient for the background."""
        if not hasattr(self, "_context"):
            return str(color)

        grad_id = f"bg_grad_{color.to_hex().replace('#', '')}"

        if grad_id not in self._gradient_cache:
            # Background fades from a slightly lighter version to the base color
            # along the light direction
            highlight = color.lighten(self.highlight_amount * 0.5)

            # Gradient direction mirrors light direction
            x1, y1 = 0.5 - self.light_direction[0] * 0.5, 0.5 - self.light_direction[1] * 0.5
            x2, y2 = 0.5 + self.light_direction[0] * 0.5, 0.5 + self.light_direction[1] * 0.5

            grad = self._context.dwg.linearGradient(
                id=grad_id,
                start=(f"{x1*100:.1f}%", f"{y1*100:.1f}%"),
                end=(f"{x2*100:.1f}%", f"{y2*100:.1f}%"),
                gradientUnits="userSpaceOnUse",  # Global gradient
            )
            grad.add_stop_color(offset="0%", color=str(highlight))
            grad.add_stop_color(offset="100%", color=str(color))

            self._context.dwg.defs.add(grad)
            self._gradient_cache[grad_id] = f"url(#{grad_id})"

        return self._gradient_cache[grad_id]

    def _get_global_fill(self, color: Color) -> str:
        """Create a global linear gradient for a large shape (e.g. unioned network)."""
        if not hasattr(self, "_context"):
            return str(color)

        grad_id = f"global_grad_{color.to_hex().replace('#', '')}"

        if grad_id not in self._gradient_cache:
            highlight = color.lighten(self.highlight_amount)
            shadow = color.darken(self.shadow_amount)

            # Gradient direction mirrors light direction
            x1, y1 = 0.5 - self.light_direction[0] * 0.5, 0.5 - self.light_direction[1] * 0.5
            x2, y2 = 0.5 + self.light_direction[0] * 0.5, 0.5 + self.light_direction[1] * 0.5

            grad = self._context.dwg.linearGradient(
                id=grad_id,
                start=(f"{x1*100:.1f}%", f"{y1*100:.1f}%"),
                end=(f"{x2*100:.1f}%", f"{y2*100:.1f}%"),
                gradientUnits="objectBoundingBox",
            )
            grad.add_stop_color(offset="0%", color=str(highlight))
            grad.add_stop_color(offset="50%", color=str(color))
            grad.add_stop_color(offset="100%", color=str(shadow))

            self._context.dwg.defs.add(grad)
            self._gradient_cache[grad_id] = f"url(#{grad_id})"

        return self._gradient_cache[grad_id]

    def _get_organ_fill(self, color: Color, info: dict[str, Any]) -> str:
        """Create a directional gradient for an organ."""
        if not hasattr(self, "_context"):
            return str(color)

        # For organs, we use a radial gradient offset toward the light
        # We account for the organ's orientation to keep the highlight consistent
        angle = info.get("angle", 0.0)
        # Round angle to limit number of gradients in cache
        angle_key = int(round(angle / 10.0) * 10) % 360
        grad_id = f"organ_grad_{color.to_hex().replace('#', '')}_{angle_key}"

        if grad_id not in self._gradient_cache:
            highlight = color.lighten(self.highlight_amount)

            # Rotate light direction into organ's local space
            rad = np.radians(-angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            local_light = np.array([
                self.light_direction[0] * cos_a - self.light_direction[1] * sin_a,
                self.light_direction[0] * sin_a + self.light_direction[1] * cos_a
            ])

            # Offset focal point toward local light direction
            fx, fy = 0.5 + local_light[0] * 0.3, 0.5 + local_light[1] * 0.3

            grad = self._context.dwg.radialGradient(
                id=grad_id,
                center=("50%", "50%"),
                r="70%",
                focal=(f"{fx*100:.1f}%", f"{fy*100:.1f}%"),
                gradientUnits="objectBoundingBox",
            )
            grad.add_stop_color(offset="0%", color=str(highlight))
            grad.add_stop_color(offset="100%", color=str(color))

            self._context.dwg.defs.add(grad)
            self._gradient_cache[grad_id] = f"url(#{grad_id})"

        return self._gradient_cache[grad_id]
