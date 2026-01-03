"""Base Pattern interface for composable generative patterns.

Semantic representation of output of Generators.

Implementations should wrap battle-tested library code for common patterns,
e.g. trees, scalar fields, vector fields

All generative patterns implement this interface, enabling:
1. Semantic queries
2. Geometry export - convert to renderable/manufacturable format
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .geometry import Polyline


@dataclass
class Pattern(ABC):
    """Interface for patterns that can be spatially queried and rendered.

    A Pattern:
    - Can be rendered to geometry (to_geometry)
    - Has a string representation for display and identification

    Patterns maintain their own internal semantic structures (e.g.,
    BranchNetwork maintains parents, depths, branch_ids as separate arrays).
    The Pattern interface is how they expose themselves to the outside world.
    """

    @abstractmethod
    def to_geometry(self) -> Polyline:
        """Convert pattern to renderable geometry."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return human-readable string representation of the pattern."""
        pass
