import numpy as np
import pytest

from bp_designs.core.geometry import Canvas
from bp_designs.patterns.network.base import BranchNetwork
from bp_designs.patterns.network.refinement import NetworkRefinementStrategy


@pytest.fixture
def simple_network():
    canvas = Canvas(coords=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
    # Create a simple line: 0 -> 1 -> 2
    # Distances: 0-1 is 10, 1-2 is 0.5
    return BranchNetwork(
        node_ids=np.array([0, 1, 2], dtype=np.int16),
        positions=np.array([[0, 0], [10, 0], [10.5, 0]], dtype=float),
        parents=np.array([-1, 0, 1], dtype=np.int16),
        timestamps=np.array([0, 1, 2], dtype=np.int16),
        canvas=canvas,
    )

def test_refinement_decimate(simple_network):
    # Decimate nodes closer than 1.0. Node 2 is 0.5 from Node 1.
    strategy = NetworkRefinementStrategy(decimate_min_distance=1.0)
    refined = strategy.apply(simple_network)

    assert refined.num_nodes == 2
    assert 2 not in refined.node_ids
    assert 1 in refined.node_ids

def test_refinement_subdivide(simple_network):
    strategy = NetworkRefinementStrategy(subdivide=True)
    refined = strategy.apply(simple_network)

    # Original had 2 segments (0-1, 1-2), subdividing adds 2 nodes
    assert refined.num_nodes == 5

def test_refinement_relocate(simple_network):
    # Relocate with alpha=1.0 should move node 1 to midpoint of 0 and 2
    # Node 0 is at (0,0), Node 2 is at (10.5, 0). Midpoint is (5.25, 0)
    strategy = NetworkRefinementStrategy(relocate_alpha=1.0, relocate_iterations=1, relocate_fix_leaves=True, relocate_fix_roots=True)
    refined = strategy.apply(simple_network)

    # Node 1 is at index 1
    np.testing.assert_array_almost_equal(refined.positions[1], [5.25, 0])

def test_refinement_combined(simple_network):
    # Decimate (removes 2) -> Subdivide (adds 1 midpoint for 0-1) -> Relocate
    strategy = NetworkRefinementStrategy(
        decimate_min_distance=1.0,
        subdivide=True,
        relocate_alpha=0.5
    )
    refined = strategy.apply(simple_network)

    # After decimate: 0, 1 (1 segment)
    # After subdivide: 0, 1, 3 (midpoint) (2 segments: 0-3, 3-1)
    assert refined.num_nodes == 3
