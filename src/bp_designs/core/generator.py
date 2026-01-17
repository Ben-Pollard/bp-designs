from abc import ABC, abstractmethod
from typing import Any

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

    @classmethod
    def create_and_generate(cls, params: dict[str, Any]) -> Pattern:
        """Helper to create generator and generate pattern by automatically
        splitting structural and rendering parameters.

        Args:
            params: Combined dictionary of structural and rendering parameters.
        """
        import inspect

        # Inspect __init__ to find structural parameters
        init_sig = inspect.signature(cls.__init__)
        structural = {}
        rendering = {}

        for k, v in params.items():
            if k in init_sig.parameters:
                structural[k] = v
            else:
                rendering[k] = v

        # Create generator with structural params
        gen = cls(**structural)

        # Generate pattern with remaining params as rendering defaults
        return gen.generate_pattern(**rendering)
