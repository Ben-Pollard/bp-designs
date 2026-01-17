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
    @classmethod
    def from_hsl(cls, h: float, s: float, lightness: float, a: float = 1.0) -> Color:
        """Create Color from HSL (0-1)."""
        r, g, b = colorsys.hls_to_rgb(h, lightness, s)
        return cls(r, g, b, a)

    def to_hex(self) -> str:
        """Convert to hex string."""
        ri = int(round(self.r * 255))
        gi = int(round(self.g * 255))
        bi = int(round(self.b * 255))
        return f"#{ri:02x}{gi:02x}{bi:02x}"

    def to_hsl(self) -> tuple[float, float, float]:
        """Convert to HSL (0-1)."""
        h, lightness, s = colorsys.rgb_to_hls(self.r, self.g, self.b)
        return h, s, lightness

    def with_hsl(
        self, h: float | None = None, s: float | None = None, lightness: float | None = None
    ) -> Color:
        """Return a new color with overridden HSL components.

        Args:
            h: Optional hue override (0-1)
            s: Optional saturation override (0-1)
            lightness: Optional lightness override (0-1)
        """
        curr_h, curr_s, curr_lightness = self.to_hsl()
        return Color.from_hsl(
            h if h is not None else curr_h,
            s if s is not None else curr_s,
            lightness if lightness is not None else curr_lightness,
            self.a,
        )

    def complementary(self) -> Color:
        """Return the complementary color using an RYB color wheel for more natural results.

        Standard HSL complementary (180 degree shift) often produces 'neon' or synthetic
        complements (e.g., Red -> Cyan). This implementation uses a warped hue wheel
        to provide artistic complements (e.g., Red -> Green, Blue -> Orange).
        """
        h, s, lightness = self.to_hsl()

        # RGB Hue to RYB Hue mapping
        # Points: (RGB Hue, RYB Hue) in degrees
        rgb_points = [0, 30, 60, 120, 240, 270, 360]
        ryb_points = [0, 60, 120, 180, 240, 300, 360]

        def interp(x, xp, fp):
            if x <= xp[0]:
                return fp[0]
            if x >= xp[-1]:
                return fp[-1]
            for i in range(len(xp) - 1):
                if xp[i] <= x <= xp[i + 1]:
                    t = (x - xp[i]) / (xp[i + 1] - xp[i])
                    return fp[i] + t * (fp[i + 1] - fp[i])
            return fp[-1]

        # Convert to degrees for easier mapping
        h_deg = h * 360.0

        # 1. RGB Hue -> RYB Hue
        ryb_h = interp(h_deg, rgb_points, ryb_points)

        # 2. Shift by 180 degrees in RYB space
        ryb_h_comp = (ryb_h + 180.0) % 360.0

        # 3. RYB Hue -> RGB Hue
        h_comp_deg = interp(ryb_h_comp, ryb_points, rgb_points)

        return Color.from_hsl(h_comp_deg / 360.0, s, lightness, self.a)

    def lighten(self, amount: float) -> Color:
        """Return a lightened version of the color."""
        h, s, lightness = self.to_hsl()
        return Color.from_hsl(h, s, min(1.0, lightness + amount), self.a)

    def darken(self, amount: float) -> Color:
        """Return a darkened version of the color."""
        h, s, lightness = self.to_hsl()
        return Color.from_hsl(h, s, max(0.0, lightness - amount), self.a)

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

    def __init__(self, colors: list[Color], name: str = "unnamed"):
        self.colors = colors
        self.name = name

    @classmethod
    def from_base(cls, base: Color, n: int = 5, variation: float = 0.1) -> Palette:
        """Create a palette by varying a base color."""
        h, s, lightness = base.to_hsl()
        colors = []
        for i in range(n):
            # Slight variation in hue and lightness
            offset = (i / max(1, n - 1) - 0.5) * variation
            colors.append(
                Color.from_hsl((h + offset / 2) % 1.0, s, max(0, min(1, lightness + offset)), base.a)
            )
        return cls(colors, name=f"Variation of {base.to_hex()}")

    def __getitem__(self, index: int) -> Color:
        return self.colors[index]

    def __len__(self) -> int:
        return len(self.colors)

    def __iter__(self):
        return iter(self.colors)
