"""Base Pattern interface for composable generative patterns.

All generative patterns implement this interface, enabling:
1. Field queries - sample pattern properties at arbitrary spatial positions
2. Geometry export - convert to renderable/manufacturable format
3. Universal composition - any two patterns can be combined
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bp_designs.patterns import Geometry


@dataclass
class Pattern(ABC):
    """Interface for patterns that can be spatially queried and rendered.

    A Pattern:
    - Can be sampled as a field at arbitrary positions (sample_field)
    - Can be rendered to geometry (to_geometry)
    - Has spatial bounds
    - Declares what channels are available

    Patterns maintain their own internal semantic structures (e.g.,
    BranchNetwork maintains parents, depths, branch_ids as separate arrays).
    The Pattern interface is how they expose themselves to the outside world.
    """

    @abstractmethod
    def sample_field(self, points: np.ndarray, channel: str) -> np.ndarray:
        """Sample this pattern as a field at given points.

        Args:
            points: (N, 2) array of query positions
            channel: Which property to sample (pattern-specific)

        Returns:
            (N,) or (N, k) array of values at each point
            Most channels return (N,) scalar values
            Some channels (like 'direction') may return (N, k) vectors

        Raises:
            ValueError: If channel is not in available_channels()

        Example:
            tree = BranchNetwork(...)
            points = np.array([[10, 20], [30, 40]])
            distances = tree.sample_field(points, 'distance')  # (2,) array
            directions = tree.sample_field(points, 'direction')  # (2, 2) array
        """
        pass

    @abstractmethod
    def available_channels(self) -> dict[str, str]:
        """Returns available channels with descriptions.

        Returns:
            dict mapping channel names to human-readable descriptions

        Example:
            {
                'distance': 'Distance to nearest branch node',
                'depth': 'Hierarchy depth from root (interpolated)',
                'density': 'Branch density (exp(-d/10))',
                'direction': 'Growth direction (returns (N,2) vectors)'
            }
        """
        pass

    @abstractmethod
    def to_geometry(self) -> Geometry:
        """Convert pattern to renderable geometry.

        Returns:
            List of polylines (ndarrays of shape (M, 2))
            Each polyline is a connected sequence of points

        Example:
            geometry = pattern.to_geometry()
            # geometry = [
            #     np.array([[0, 0], [1, 1], [2, 2]]),  # Polyline 1
            #     np.array([[0, 0], [1, -1]]),         # Polyline 2
            # ]
        """
        pass

    @abstractmethod
    def bounds(self) -> tuple[float, float, float, float]:
        """Pattern bounding box.

        Returns:
            (xmin, ymin, xmax, ymax)
        """
        pass
