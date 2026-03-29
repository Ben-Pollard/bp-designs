import numpy as np

from bp_designs.core.field import ConstantField
from bp_designs.core.geometry import Canvas
from bp_designs.generators.flow.generator import FlowGenerator
from bp_designs.generators.flow.strategies import EulerIntegrator, FixedLengthTermination, RandomSeeding
from bp_designs.patterns.flow import StreamlinePattern


def test_flow_generator_basic():
    canvas = Canvas.from_width_height(100, 100)
    field = ConstantField(np.array([1.0, 0.0]))

    seeding = RandomSeeding(np.array([[0.0, 0.0], [100.0, 100.0]]), num_seeds=10)
    integrator = EulerIntegrator()
    termination = FixedLengthTermination(max_steps=5)

    generator = FlowGenerator(
        canvas=canvas,
        field=field,
        seeding_strategy=seeding,
        integration_strategy=integrator,
        termination_strategy=termination,
        dt=1.0
    )

    pattern = generator.generate_pattern()

    assert isinstance(pattern, StreamlinePattern)
    assert len(pattern.streamlines) == 10
    for streamline in pattern.streamlines:
        # 5 steps + 1 seed point = 6 points
        assert len(streamline) == 6
        # Constant field [1, 0] means x increases, y stays same
        assert np.allclose(np.diff(streamline[:, 0]), 1.0)
        assert np.allclose(np.diff(streamline[:, 1]), 0.0)

def test_streamline_pattern_to_geometry():
    canvas = Canvas.from_width_height(100, 100)
    streamlines = [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[10.0, 10.0], [11.0, 11.0]])
    ]
    pattern = StreamlinePattern(streamlines=streamlines)

    geometry = pattern.to_geometry(canvas)
    from bp_designs.core.geometry import Polyline
    assert isinstance(geometry, Polyline)
    assert len(geometry.polylines) == 2
    assert np.allclose(geometry.polylines[0], streamlines[0])

def test_streamline_simplification():
    # A line with a middle point that is exactly on the segment
    streamline = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    pattern = StreamlinePattern(streamlines=[streamline])

    # With epsilon=0.1, the middle point should be removed
    geom = pattern.to_geometry(epsilon=0.1)
    assert len(geom.polylines[0]) == 2
    assert np.allclose(geom.polylines[0], [[0.0, 0.0], [10.0, 0.0]])
