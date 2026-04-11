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
        """Map (N, 2) positions to (N, 2) vectors using noise.

        Args:
            positions: (N, 2) array of coordinates.

        Returns:
            (N, 2) array of vectors.
        """
        x = positions[:, 0] / self.scale
        y = positions[:, 1] / self.scale

        # vnoise.noise2 returns a grid if x and y are arrays.
        # For point-wise noise, a list comprehension is actually faster than
        # computing the full N x N grid and taking the diagonal.
        n_vals = np.array([self.noise.noise2(px, py) for px, py in zip(x, y)])

        angles = n_vals * 2 * np.pi
        vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        return vectors * self.strength

class AngleField(ConstantField):
    """A field that returns a constant vector at a specific angle."""
    def __init__(self, angle: float, degrees: bool = True):
        rad = np.radians(angle) if degrees else angle
        vector = np.array([np.cos(rad), np.sin(rad)])
        super().__init__(vector)

class VortexField(Field):
    """A field where vectors rotate around a center."""
    def __init__(self, center: np.ndarray = np.array([0.0, 0.0]), strength: float = 1.0, clockwise: bool = True):
        self.center = np.asanyarray(center)
        self.strength = strength
        self.clockwise = clockwise

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        diff = positions - self.center
        # Rotate 90 degrees: (x, y) -> (-y, x) for counter-clockwise, (y, -x) for clockwise
        if self.clockwise:
            vortex = np.stack([diff[:, 1], -diff[:, 0]], axis=1)
        else:
            vortex = np.stack([-diff[:, 1], diff[:, 0]], axis=1)

        dist = np.linalg.norm(vortex, axis=1, keepdims=True)
        dist = np.linalg.norm(vortex, axis=1, keepdims=True)
        dist = np.where(dist == 0, 1.0, dist)
        return (vortex / dist) * self.strength

class WorleyField(Field):
    """A field based on Worley (Cellular) noise."""
    def __init__(self, seed: int = 0, num_points: int = 20, strength: float = 1.0, bounds: np.ndarray = np.array([[0, 0], [200, 200]])):
        self.rng = np.random.default_rng(seed)
        self.points = self.rng.uniform(bounds[0], bounds[1], (num_points, 2))
        self.strength = strength

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        from scipy.spatial import cKDTree
        tree = cKDTree(self.points)
        dists, indices = tree.query(positions)

        # Use the distance to the nearest point to drive an angle
        # This creates the "shattered" cell look
        angles = dists * 0.1 * 2 * np.pi
        vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        return vectors * self.strength

class ValueNoiseField(Field):
    """A field based on Value noise (interpolated random scalars)."""
    def __init__(self, seed: int = 0, scale: float = 20.0, strength: float = 1.0, grid_res: int = 10):
        self.rng = np.random.default_rng(seed)
        self.scale = scale
        self.strength = strength
        self.grid_res = grid_res
        self.grid = self.rng.uniform(-1, 1, (grid_res, grid_res))

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        from scipy.interpolate import RegularGridInterpolator
        x = np.linspace(0, 200, self.grid_res)
        y = np.linspace(0, 200, self.grid_res)
        interp = RegularGridInterpolator((x, y), self.grid, bounds_error=False, fill_value=0)

        n_vals = interp(positions)
        angles = n_vals * 2 * np.pi
        vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        return vectors * self.strength

class SineWaveField(Field):
    """A field based on summed sine waves (Plasma)."""
    def __init__(self, num_waves: int = 3, scale: float = 50.0, strength: float = 1.0, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.strength = strength
        self.scale = scale
        # Random directions and phases for the waves
        self.angles = self.rng.uniform(0, 2 * np.pi, num_waves)
        self.phases = self.rng.uniform(0, 2 * np.pi, num_waves)

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        n_vals = np.zeros(len(positions))
        for angle, phase in zip(self.angles, self.phases):
            # Project positions onto the wave direction
            dir_vec = np.array([np.cos(angle), np.sin(angle)])
            projection = positions @ dir_vec
            n_vals += np.sin(projection / self.scale + phase)

        n_vals /= len(self.angles)
        angles = n_vals * 2 * np.pi
        vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        return vectors * self.strength

class WaveletField(Field):
    """A field based on a simplified Wavelet noise (band-limited)."""
    def __init__(self, seed: int = 0, scale: float = 30.0, strength: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.scale = scale
        self.strength = strength
        # Create a small grid of random coefficients
        self.coeffs = self.rng.uniform(-1, 1, (8, 8))

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        # Simplified wavelet: use a small grid and bicubic-like interpolation
        # to ensure band-limiting (no high frequency noise)
        from scipy.interpolate import RegularGridInterpolator
        x = np.linspace(0, 200, 8)
        y = np.linspace(0, 200, 8)
        interp = RegularGridInterpolator((x, y), self.coeffs, method='cubic', bounds_error=False, fill_value=0)

        n_vals = interp(positions)
        angles = n_vals * 2 * np.pi
        vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        return vectors * self.strength
