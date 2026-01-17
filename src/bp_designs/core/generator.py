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
        """Generate a pattern.

        Args:
            **kwargs: Rendering parameters to be stored in the resulting Pattern.

        Returns:
            Pattern instance (specific type depends on generator)
        """
        pass

    def __call__(self, **kwargs) -> Pattern:
        """Generate pattern - alias for generate_pattern() for cleaner syntax."""
        return self.generate_pattern(**kwargs)
