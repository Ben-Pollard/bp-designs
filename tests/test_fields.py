
import numpy as np
from bp_designs.core.field import ConstantField, NoiseField, RadialField


def test_constant_field():
    direction = np.array([1.0, 0.0])
    field = ConstantField(direction)
    positions = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, -5.0]])
    vectors = field(positions)

    assert vectors.shape == (3, 2)
    assert np.allclose(vectors, direction)

def test_radial_field_outward():
    center = np.array([0.0, 0.0])
    field = RadialField(center, strength=1.0)

    # Point at (1, 0) should have vector (1, 0)
    pos = np.array([[1.0, 0.0]])
    vectors = field(pos)
    assert np.allclose(vectors, [[1.0, 0.0]])

    # Point at (0, 2) should have vector (0, 1) (normalized)
    pos = np.array([[0.0, 2.0]])
    vectors = field(pos)
    assert np.allclose(vectors, [[0.0, 1.0]])

def test_field_addition():
    f1 = ConstantField(np.array([1.0, 0.0]))
    f2 = ConstantField(np.array([0.0, 1.0]))
    combined = f1 + f2

    pos = np.array([[0.0, 0.0]])
    vectors = combined(pos)
    assert np.allclose(vectors, [[1.0, 1.0]])

def test_noise_field_determinism():
    seed = 42
    f1 = NoiseField(seed=seed, scale=10.0)
    f2 = NoiseField(seed=seed, scale=10.0)

    pos = np.random.rand(10, 2)
    assert np.allclose(f1(pos), f2(pos))

def test_noise_field_different_seeds():
    f1 = NoiseField(seed=42)
    f2 = NoiseField(seed=43)

    pos = np.random.rand(10, 2)
    assert not np.allclose(f1(pos), f2(pos))
