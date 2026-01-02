from abc import ABC, abstractmethod

from .pattern import Pattern


class Generator(ABC):
    """Base class for pattern generators.

    A Generator:
    - Encapsulates an algorithm (space colonization, Voronoi, etc.)
    - Has parameters that control generation
    - Produces Pattern instances via __call__()
    - Is deterministic (same seed → same output)

    Generators may optionally accept guidance from other patterns.
    """

    @abstractmethod
    def generate_pattern(self, **kwargs) -> Pattern:
        """Generate a pattern, optionally guided by based on inputs from previously
        generated patterns.

        Args:
            **kwargs: Generator-specific Pattern of Geometry params. Other params go in __init__.

        Returns:
            Pattern instance (specific type depends on generator)

        Example:
            # Unguided generation
            gen = SpaceColonization(seed=42)
            tree = gen.generate_pattern(root=Point(100, 50))

            # Guided generation
            voronoi = VoronoiTessellation(seed=42).generate_pattern()
            tree = gen.generate_pattern(
                intial_boundary: Polygon = voronoi.get_cell(0).to_Polygon()
                final_boundary: Polygon = voronoi.to_Polygon()
            )
        """
        pass

    def __call__(self, **kwargs) -> Pattern:
        """Generate pattern - alias for generate_pattern() for cleaner syntax."""
        return self.generate_pattern(**kwargs)
