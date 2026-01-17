"""Color utility for generative patterns.

Provides HSL/RGB/Hex conversions and color relationship logic (complementary, palettes).
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class Color:
    """Representation of a color with conversion and relationship methods.

    Stored internally as RGB (0-1).
    """

    r: float
    g: float
    b: float
    a: float = 1.0

    @classmethod
    def from_hex(cls, hex_str: str) -> Color:
        """Create Color from hex string (e.g., '#ff0000' or 'ff0000')."""
        hex_str = hex_str.lstrip("#")
        if len(hex_str) == 3:
            hex_str = "".join([c * 2 for c in hex_str])
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        return cls(r, g, b)

    @classmethod
    def from_rgb(cls, r: int, g: int, b: int, a: float = 1.0) -> Color:
        """Create Color from RGB integers (0-255)."""
        return cls(r / 255.0, g / 255.0, b / 255.0, a)

    @classmethod
    def from_hsl(cls, h: float, s: float, l: float, a: float = 1.0) -> Color:
        """Create Color from HSL (0-1)."""
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return cls(r, g, b, a)

    def to_hex(self) -> str:
        """Convert to hex string."""
        ri = int(round(self.r * 255))
        gi = int(round(self.g * 255))
        bi = int(round(self.b * 255))
        return f"#{ri:02x}{gi:02x}{bi:02x}"

    def to_hsl(self) -> tuple[float, float, float]:
        """Convert to HSL (0-1)."""
        h, l, s = colorsys.rgb_to_hls(self.r, self.g, self.b)
        return h, s, l

    def complementary(self) -> Color:
        """Return the complementary color (180 degree hue shift)."""
        h, s, l = self.to_hsl()
        return Color.from_hsl((h + 0.5) % 1.0, s, l, self.a)

    def lighten(self, amount: float) -> Color:
        """Return a lightened version of the color."""
        h, s, l = self.to_hsl()
        return Color.from_hsl(h, s, min(1.0, l + amount), self.a)

    def darken(self, amount: float) -> Color:
        """Return a darkened version of the color."""
        h, s, l = self.to_hsl()
        return Color.from_hsl(h, s, max(0.0, l - amount), self.a)

    @classmethod
    def lerp(cls, c1: Color, c2: Color, t: float) -> Color:
        """Linearly interpolate between two colors."""
        t = max(0.0, min(1.0, t))
        return cls(
            r=c1.r + (c2.r - c1.r) * t,
            g=c1.g + (c2.g - c1.g) * t,
            b=c1.b + (c2.b - c1.b) * t,
            a=c1.a + (c2.a - c1.a) * t,
        )

    def jitter(self, amount: float, rng: np.random.Generator | None = None) -> Color:
        """Return a color with random variation in RGB components."""
        import numpy as np

        if rng is None:
            rng = np.random.default_rng()
        v = rng.uniform(-amount, amount, 3)
        return Color(
            r=max(0.0, min(1.0, self.r * (1 + v[0]))),
            g=max(0.0, min(1.0, self.g * (1 + v[1]))),
            b=max(0.0, min(1.0, self.b * (1 + v[2]))),
            a=self.a,
        )

    def __str__(self) -> str:
        return self.to_hex()


class Palette:
    """Collection of related colors."""

    def __init__(self, base: Color):
        self.base = base

    def get_background(self) -> Color:
        """Get a suitable background color (complementary and lightened)."""
        return self.base.complementary().lighten(0.4)

    def get_range(self, n: int, variation: float = 0.1) -> list[Color]:
        """Get a range of colors based on the base color."""
        h, s, l = self.base.to_hsl()
        colors = []
        for i in range(n):
            # Slight variation in hue and lightness
            offset = (i / max(1, n - 1) - 0.5) * variation
            colors.append(Color.from_hsl((h + offset / 2) % 1.0, s, max(0, min(1, l + offset)), self.base.a))
        return colors
