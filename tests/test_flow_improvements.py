import numpy as np
import pytest

from bp_designs.core.field import ConstantField, NoiseField
from bp_designs.core.geometry import Canvas
from bp_designs.generators.flow.generator import FlowGenerator
from bp_designs.generators.flow.strategies import (
    EulerIntegrator,
    FixedLengthTermination,
    ProximityTermination,
    RandomSeeding,
)
from bp_designs.patterns.flow import (
    MagnitudeColor,
    MagnitudeWidth,
)


def test_vectorized_noise_field():
    field = NoiseField(scale=10.0)
    positions = np.random.rand(10, 2)
    vectors = field(positions)
    assert vectors.shape == (10, 2)
    # Check that it's not all zeros
    assert np.any(vectors != 0)


def test_dynamic_proximity_termination():
    canvas = Canvas.from_width_height(100, 100)
    # Field that points right
    field = ConstantField(np.array([1.0, 0.0]))

    # Seed at (10, 50)
    seeds = np.array([[10.0, 50.0]])
    seeding = RandomSeeding(np.array([[0, 0], [100, 100]]), num_seeds=1)
    seeding.generate = lambda: seeds

    # Existing point at (20, 50)
    existing = np.array([[20.0, 50.0]])
    # min_dist must be smaller than dt (1.0) to avoid self-termination
    termination = ProximityTermination(existing, min_dist=0.5)

    gen = FlowGenerator(
        canvas=canvas,
        field=field,
        seeding_strategy=seeding,
        integration_strategy=EulerIntegrator(),
        termination_strategy=termination,
        dt=1.0
    )

    pattern = gen.generate_pattern()
    # Should stop near (20, 50)
    # Steps: (10,50) -> (11,50) -> ... -> (18,50) -> (19,50) -> (20,50) [STOP]
    # The last point added should be within 2.0 of (20,50)
    # With dt=1.0 and field=(1,0), it moves 1 unit per step.
    # (10,50) -> (11,50) -> ... -> (18,50) -> (19,50)
    # At (18,50), next is (19,50). dist(19, 20) = 1.0 < 2.0.
    # So (19,50) should be the last point if it terminates BEFORE adding the point,
    # or it adds the point and then terminates.
    # In our implementation:
    # next_positions = ... (19,50)
    # should_stop = ... (True because dist((19,50), (20,50)) = 1.0 < 2.0)
    # streamlines[idx].append(new_pos) -> adds (19,50)
    # if not stop: ... (False, so doesn't update current_positions or still_active)

    last_point = pattern.streamlines[0][-1]
    assert np.linalg.norm(last_point - existing[0]) <= 0.5
    assert len(pattern.streamlines[0]) <= 12


def test_magnitude_mapping():
    # Field with varying magnitude: Radial field
    from bp_designs.core.field import RadialField
    field = RadialField(center=np.array([0.0, 0.0]), strength=1.0)
    # At distance r, magnitude is 1.0 (normalized in RadialField implementation)
    # Wait, RadialField in field.py: (diff / dist) * self.strength
    # So magnitude is always self.strength.

    # Let's use a custom field for varying magnitude
    class VaryingField(ConstantField):
        def __call__(self, positions):
            # Magnitude = x coordinate
            mags = positions[:, 0:1]
            return np.hstack([mags, np.zeros_like(mags)])

    canvas = Canvas.from_width_height(100, 100)
    field = VaryingField(np.array([1.0, 0.0]))
    seeding = RandomSeeding(np.array([[10, 50], [10, 50]]), num_seeds=1)

    gen = FlowGenerator(
        canvas=canvas,
        field=field,
        seeding_strategy=seeding,
        integration_strategy=EulerIntegrator(),
        termination_strategy=FixedLengthTermination(max_steps=5),
        dt=1.0
    )

    pattern = gen.generate_pattern()
    assert len(pattern.magnitudes) == 1
    assert len(pattern.magnitudes[0]) == len(pattern.streamlines[0])
    # Magnitudes should be increasing as x increases
    assert np.all(np.diff(pattern.magnitudes[0]) >= 0)

    # Test strategies
    width_strat = MagnitudeWidth(min_width=1.0, max_width=5.0, mag_range=(10.0, 20.0))
    color_strat = MagnitudeColor(color1="#0000ff", color2="#ff0000", mag_range=(10.0, 20.0))

    # Check width at start (mag=10)
    w_start = width_strat.get_width(pattern.streamlines[0], 0, pattern.magnitudes[0])
    assert w_start == pytest.approx(1.0)

    # Check color at start
    c_start = color_strat.get_color(pattern.streamlines[0], 0, pattern.magnitudes[0])
    assert c_start.lower() == "#0000ff"
