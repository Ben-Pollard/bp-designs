from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from vnoise import vnoise


class Field(ABC):
    """Abstract base class for vector fields."""

    @abstractmethod
    def __call__(self, positions: np.ndarray) -> np.ndarray:
        """Map (N, 2) positions to (N, 2) vectors.
        
        Args:
            positions: (N, 2) array of coordinates.
            
        Returns:
            (N, 2) array of vectors.
        """
        pass

    def __add__(self, other: Field) -> Field:
        return CompositeField(self, other, np.add)

    def __sub__(self, other: Field) -> Field:
        return CompositeField(self, other, np.subtract)

    def __mul__(self, other: Field | float | np.ndarray) -> Field:
        if isinstance(other, Field):
            return CompositeField(self, other, np.multiply)
        return ScaledField(self, other)

class CompositeField(Field):
    """A field composed of two other fields via an operator."""
    def __init__(self, f1: Field, f2: Field, op: callable):
        self.f1 = f1
        self.f2 = f2
        self.op = op

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        return self.op(self.f1(positions), self.f2(positions))

class ScaledField(Field):
    """A field scaled by a constant or array."""
    def __init__(self, field: Field, scale: float | np.ndarray):
        self.field = field
        self.scale = scale

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        return self.field(positions) * self.scale

class ConstantField(Field):
    """A field that returns a constant vector everywhere."""
    def __init__(self, vector: np.ndarray):
        self.vector = np.asanyarray(vector)

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        # Return the same vector for all N positions
        return np.tile(self.vector, (len(positions), 1))

class RadialField(Field):
    """A field where vectors point away from or towards a center."""
    def __init__(self, center: np.ndarray = np.array([0.0, 0.0]), strength: float = 1.0):
        self.center = np.asanyarray(center)
        self.strength = strength

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        diff = positions - self.center
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        # Avoid division by zero at the center
        dist = np.where(dist == 0, 1.0, dist)
        return (diff / dist) * self.strength

class NoiseField(Field):
    """A field based on Perlin/Simplex noise."""
    def __init__(self, seed: int = 0, scale: float = 10.0, strength: float = 1.0):
        self.noise = vnoise.Noise(seed=seed)
        self.scale = scale
        self.strength = strength

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        # vnoise.noise2 can take arrays for x and y
        # It returns a grid if x and y are arrays, but we want point-wise
        # Actually, let's check if it supports point-wise or if we need to loop/vectorize differently
        x = positions[:, 0] / self.scale
        y = positions[:, 1] / self.scale

        # If noise2 returns a grid, we might need to use a different approach or just loop if N is small,
        # but for performance we want vectorization.
        # Based on my test: noise.noise2(np.array([0.1, 0.2]), np.array([0.3, 0.4])) returned a 2x2 grid.
        # We want the diagonal: [noise(0.1, 0.3), noise(0.2, 0.4)]

        # To get point-wise noise efficiently without a full grid:
        # Some libraries support this, vnoise might not directly.
        # Let's use a list comprehension for now if it's not directly supported,
        # or check if there's a better way.

        n_vals = np.array([self.noise.noise2(px, py) for px, py in zip(x, y)])
        angles = n_vals * 2 * np.pi

        vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        return vectors * self.strength
