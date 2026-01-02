"""2D primitive shape generators.

Generates basic geometric shapes as ShapePattern instances.
"""

from __future__ import annotations

from bp_designs.core.generator import Generator


class Primitive2D(Generator):
    """Generate 2D primitive shapes as patterns.

    Supports circles, rectangles, regular polygons, and arbitrary polygons.

    This only needs to know about how to create the semantic information for a shape

    ShapePattern holds the semantic information and knows how to create a Geometry from it.

    For something simple like a circle that could just be passing the centre Point and radius
    through the Pattern
    """
