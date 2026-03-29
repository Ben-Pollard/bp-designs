import numpy as np

from bp_designs.core.field import ConstantField, Field
from bp_designs.generators.flow.strategies import EulerIntegrator, RK4Integrator


def test_euler_integrator_constant():
    field = ConstantField(np.array([1.0, 0.0]))
    integrator = EulerIntegrator()
    pos = np.array([[0.0, 0.0]])
    dt = 1.0

    next_pos = integrator.step(field, pos, dt)
    assert np.allclose(next_pos, [[1.0, 0.0]])

def test_rk4_integrator_constant():
    field = ConstantField(np.array([1.0, 0.0]))
    integrator = RK4Integrator()
    pos = np.array([[0.0, 0.0]])
    dt = 1.0

    next_pos = integrator.step(field, pos, dt)
    assert np.allclose(next_pos, [[1.0, 0.0]])

def test_integrator_drift_comparison():
    # Circular field: v = (-y, x)
    class CircularField(Field):
        def __call__(self, positions: np.ndarray) -> np.ndarray:
            return np.stack([-positions[:, 1], positions[:, 0]], axis=1)

    field = CircularField()
    start_pos = np.array([[1.0, 0.0]])
    dt = 0.1
    steps = 100

    # Euler
    euler = EulerIntegrator()
    pos_e = start_pos.copy()
    for _ in range(steps):
        pos_e = euler.step(field, pos_e, dt)

    # RK4
    rk4 = RK4Integrator()
    pos_r = start_pos.copy()
    for _ in range(steps):
        pos_r = rk4.step(field, pos_r, dt)

    # In a circular field, the radius should stay 1.0.
    # Euler always drifts outwards. RK4 is much more stable.
    dist_e = np.linalg.norm(pos_e[0])
    dist_r = np.linalg.norm(pos_r[0])

    assert dist_e > 1.0
    assert abs(dist_r - 1.0) < abs(dist_e - 1.0)

from bp_designs.core.geometry import Polygon
from bp_designs.generators.flow.strategies import (
    BoundaryTermination,
    FixedLengthTermination,
    GridSeeding,
    PoissonDiscSeeding,
    ProximityTermination,
    RandomSeeding,
)


def test_random_seeding():
    bounds = np.array([[0.0, 0.0], [10.0, 10.0]])
    seeding = RandomSeeding(bounds, num_seeds=100, seed=42)
    seeds = seeding.generate()

    assert seeds.shape == (100, 2)
    assert np.all(seeds >= 0.0)
    assert np.all(seeds <= 10.0)

def test_grid_seeding():
    bounds = np.array([[0.0, 0.0], [2.0, 2.0]])
    seeding = GridSeeding(bounds, resolution=(3, 3))
    seeds = seeding.generate()

    # 3x3 grid = 9 points
    assert seeds.shape == (9, 2)
    # Check corners
    assert any(np.allclose(s, [0.0, 0.0]) for s in seeds)
    assert any(np.allclose(s, [2.0, 2.0]) for s in seeds)

def test_poisson_disc_seeding():
    bounds = np.array([[0.0, 0.0], [10.0, 10.0]])
    min_dist = 2.0
    seeding = PoissonDiscSeeding(bounds, min_dist=min_dist, seed=42)
    seeds = seeding.generate()

    assert len(seeds) > 0
    # Check minimum distance between any two seeds
    from scipy.spatial.distance import pdist
    if len(seeds) > 1:
        distances = pdist(seeds)
        assert np.all(distances >= min_dist * 0.99) # Allow small float tolerance

def test_fixed_length_termination():
    term = FixedLengthTermination(max_steps=10)
    # Should not terminate before 10 steps
    assert not term.should_terminate(np.array([[0.0, 0.0]]), step_count=5)
    # Should terminate at 10 steps
    assert term.should_terminate(np.array([[0.0, 0.0]]), step_count=10)

def test_boundary_termination():
    poly = Polygon(coords=np.array([[0, 0], [10, 0], [10, 10], [0, 10]]))
    term = BoundaryTermination(poly)

    assert not term.should_terminate(np.array([[5.0, 5.0]]), step_count=0)
    assert term.should_terminate(np.array([[15.0, 5.0]]), step_count=0)

def test_proximity_termination():
    existing_points = np.array([[0.0, 0.0], [10.0, 10.0]])
    term = ProximityTermination(existing_points, min_dist=1.0)

    # Far from existing points
    assert not term.should_terminate(np.array([[5.0, 5.0]]), step_count=0)
    # Close to (0, 0)
    assert term.should_terminate(np.array([[0.5, 0.5]]), step_count=0)
