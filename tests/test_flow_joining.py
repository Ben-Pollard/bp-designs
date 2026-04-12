import numpy as np

from bp_designs.core.field import Field
from bp_designs.core.geometry import Canvas
from bp_designs.generators.flow.generator import FlowConfig, FlowGenerator
from bp_designs.generators.flow.strategies import (
    AngleJoinStrategy,
    EulerIntegrator,
    ProximityTermination,
    SeedingStrategy,
)
from bp_designs.patterns.flow import TaperedWidth


class ConstantField(Field):
    def __init__(self, velocity: np.ndarray):
        self.velocity = velocity

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        return np.tile(self.velocity, (len(positions), 1))


class ManualSeeding(SeedingStrategy):
    def __init__(self, seeds: list[list[float]]):
        self.seeds = np.array(seeds)

    def generate(self) -> np.ndarray:
        return self.seeds


def test_proximity_termination_metadata():
    """Test that get_collision_metadata handles query points correctly."""
    existing_points = np.array([[10.0, 10.0], [20.0, 20.0]])
    existing_ids = np.array([1, 2])
    existing_indices = np.array([0, 0])

    term = ProximityTermination(
        existing_points=existing_points,
        min_dist=5.0,
        existing_ids=existing_ids,
        existing_indices=existing_indices,
    )

    # Query with a point close to the first existing point
    query_pos = np.array([[11.0, 11.0]])
    metadata = term.get_collision_metadata(query_pos, ids=np.array([3]))

    assert len(metadata) == 1
    assert metadata[0] is not None
    assert metadata[0]["id"] == 1
    assert metadata[0]["index"] == 0


def test_angle_join_strategy_alignment():
    """Test the AngleJoinStrategy logic for alignment and bridge segments."""
    strategy = AngleJoinStrategy(max_angle_deg=30.0, endpoint_only=True)

    # Target streamline: horizontal line from (10, 10) to (20, 10)
    target_streamline = np.array([[10.0, 10.0], [20.0, 10.0]])

    # Case 1: Joining to the end (index 1), aligned and approaching
    current_pos = np.array([19.5, 10.0])
    current_dir = np.array([1.0, 0.0])
    meta = {"id": 1, "index": 1, "dist": 0.5}
    assert strategy.should_join(current_pos, current_dir, meta, target_streamline)

    # Case 2: Joining to the end, but at a sharp angle
    current_dir = np.array([0.0, 1.0])  # Moving up
    assert not strategy.should_join(current_pos, current_dir, meta, target_streamline)


def test_flow_joining_logic():
    """Test that the generator correctly merges streamlines."""
    canvas = Canvas.from_width_height(100, 100)

    # Seed 0 at (0,0) moving right. Seed 1 at (20,0) moving right.
    # min_dist=5.0.
    class MixedField(Field):
        def __call__(self, positions: np.ndarray) -> np.ndarray:
            return np.full_like(positions, [10.0, 0.0])

    seeding = ManualSeeding([[0.0, 0.0], [20.0, 0.0]])
    field = MixedField()
    integrator = EulerIntegrator()
    term = ProximityTermination(existing_points=np.empty((0, 2)), min_dist=5.0)
    join = AngleJoinStrategy(max_angle_deg=45.0)
    config = FlowConfig(dt=1.0, max_steps=5, join_strategy=join)

    gen = FlowGenerator(
        canvas=canvas,
        field=field,
        seeding_strategy=seeding,
        integration_strategy=integrator,
        termination_strategy=term,
        config=config,
    )

    pattern = gen.generate_pattern()

    # Should have 1 merged streamline
    assert len(pattern.streamlines) == 1
    streamline = pattern.streamlines[0]

    # S1 starts at (20,0), moves to (30,0), (40,0), (50,0), (60,0), (70,0)
    # S0 starts at (0,0), moves to (10,0), then next step is (20,0)
    # (20,0) is close to S1's start point (20,0).
    # Merged: (0,0), (10,0), (20,0), (30,0), (40,0), (50,0), (60,0), (70,0)
    # Total 18 points (2 source + 10 bridge + 6 target).
    assert len(streamline) == 18
    # Source points [0,1], Bridge [2-11], Target [12-17]
    assert np.allclose(streamline[12], [20.0, 0.0])
    assert np.allclose(streamline[-1], [70.0, 0.0])


def test_flow_joining_tapered_width_continuity():
    """Test that tapered width is smooth across a join."""
    canvas = Canvas.from_width_height(100, 100)
    seeding = ManualSeeding([[0.0, 0.0], [20.0, 0.0]])

    class SimpleField(Field):
        def __call__(self, positions: np.ndarray) -> np.ndarray:
            return np.full_like(positions, [10.0, 0.0])

    gen = FlowGenerator(
        canvas=canvas,
        field=SimpleField(),
        seeding_strategy=seeding,
        integration_strategy=EulerIntegrator(),
        termination_strategy=ProximityTermination(existing_points=np.empty((0, 2)), min_dist=5.0),
        config=FlowConfig(dt=1.0, max_steps=5, join_strategy=AngleJoinStrategy()),
    )

    pattern = gen.generate_pattern()
    assert len(pattern.streamlines) == 1
    streamline = pattern.streamlines[0]

    taper = TaperedWidth(min_width=1.0, max_width=10.0)
    widths = [taper.get_width(streamline, i) for i in range(len(streamline))]

    # Widths should be strictly monotonic (decreasing from seed to tip)
    for i in range(len(widths) - 1):
        assert widths[i] > widths[i + 1]


def test_flow_steering_alignment():
    """Test that steering pulls particles towards target streamlines."""
    canvas = Canvas.from_width_height(100, 100)

    # S0 at (0,0) moving right. S1 at (5, 2) moving right.
    # Steering radius = 5.0.

    seeding = ManualSeeding([[0.0, 0.0], [5.0, 2.0]])

    class SimpleField(Field):
        def __call__(self, positions: np.ndarray) -> np.ndarray:
            return np.full_like(positions, [5.0, 0.0])

    config = FlowConfig(
        dt=1.0,
        max_steps=10,
        steering_radius=5.0,
        steering_strength=1.0,
        min_dist=1.0,
        join_strategy=AngleJoinStrategy(),
    )

    gen = FlowGenerator(
        canvas=canvas,
        field=SimpleField(),
        seeding_strategy=seeding,
        integration_strategy=EulerIntegrator(),
        termination_strategy=ProximityTermination(existing_points=np.empty((0, 2)), min_dist=1.0),
        config=config,
    )

    pattern = gen.generate_pattern()
    # S0 should steer towards S1 and eventually join it.
    assert len(pattern.streamlines) == 1
    streamline = pattern.streamlines[0]

    # The y-coordinate of S0 should increase as it approaches S1
    has_steered = False
    for p in streamline:
        if p[1] > 0.1:
            has_steered = True
            break
    assert has_steered, "Streamline should have steered towards y=2.0"
