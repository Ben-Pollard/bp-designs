from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from pattern import Pattern


class Generator(ABC):
    """Base class for pattern generators.

    A Generator:
    - Encapsulates an algorithm (space colonization, Voronoi, etc.)
    - Has parameters that control generation
    - Produces Pattern instances via generate_pattern()
    - Is deterministic (same seed → same output)

    Generators may optionally accept guidance from other patterns.
    """

    @abstractmethod
    def generate_pattern(
        self, guidance_field: Callable[[np.ndarray, str], np.ndarray] | None = None, **kwargs
    ) -> Pattern:
        """Generate a pattern, optionally guided by external field.

        Args:
            guidance_field: Optional field function(points, channel) -> values
                          If provided, influences generation behavior
            **kwargs: Generator-specific parameters (e.g., guidance_channel,
                     guidance_strength, max_iterations, etc.)

        Returns:
            Pattern instance (specific type depends on generator)

        Example:
            # Unguided generation
            gen = SpaceColonization(seed=42)
            tree = gen.generate_pattern()

            # Guided generation
            voronoi = VoronoiTessellation(seed=42).generate_pattern()
            tree = gen.generate_pattern(
                guidance_field=voronoi.sample_field,
                guidance_channel='boundary_distance',
                guidance_strength=0.5
            )
        """
        pass
