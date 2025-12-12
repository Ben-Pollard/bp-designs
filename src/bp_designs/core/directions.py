from dataclasses import dataclass

import numpy as np


@dataclass
class PairwiseCoordinateVectors:
    """Represents vectors between pairs of coordinates.

    Attributes:
        vectors: (N,M,2) array of direction vectors between coordinates (2,M) and (2,N)
        norms: (N,M) array of distances between each pair of coordinates
        directions: (N,M,2) array of normalized direction vectors (unit vectors)
    """

    vectors: np.ndarray  # (N,M,2) - direction vectors from nodes to attractions
    norms: np.ndarray  # (N,M) - distances
    directions: np.ndarray  # (N,M,2) - normalized direction vectors

    def get_n(self, n):
        return PairwiseCoordinateVectors(
            vectors=self.vectors[n], norms=self.norms[n], directions=self.directions[n]
        )


@dataclass
class DirectionVectors:
    """Represents vectors of a sequence of coordinates.

    Attributes:
        vectors: (N,2) array of direction vectors for coordinates (N,2)
        norms: (N,) array of distances of each vector
        directions: (N,2) array of normalized direction vectors (unit vectors)
    """

    vectors: np.ndarray  # (N,2) - direction vectors
    norms: np.ndarray  # (N,) - distances
    directions: np.ndarray  # (N,2) - normalized direction vectors
