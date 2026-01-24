import numpy as np
import pytest

from bp_designs.patterns.network.base import BranchNetwork
from bp_designs.patterns.network.thickness import (
    DescendantThickness,
    HierarchyThickness,
    TimestampThickness,
)


@pytest.fixture
def simple_network():
    """Create a simple Y-shaped network.
    0 (root, t=0) -> 1 (t=1) -> 2 (leaf, t=2)
                           -> 3 (leaf, t=3)
    """
    network = BranchNetwork()
    network.node_ids = [0, 1, 2, 3]
    network.parents = [-1, 0, 1, 1]
    network.timestamps = np.array([0, 1, 2, 3])
    network.depths = np.array([0, 1, 2, 2])
    network.positions = np.zeros((4, 2))
    return network

def test_timestamp_thickness_range(simple_network):
    min_t, max_t = 1.0, 10.0
    # power=1.0 (linear)
    strategy = TimestampThickness(min_thickness=min_t, max_thickness=max_t, power=1.0)
    thickness = strategy.compute_thickness(simple_network)

    assert thickness.min() == pytest.approx(min_t)
    assert thickness.max() == pytest.approx(max_t)
    # Node 0 (t=0) should be thickest, Node 3 (t=3) thinnest
    assert thickness[0] == max_t
    assert thickness[3] == min_t

def test_timestamp_thickness_power(simple_network):
    min_t, max_t = 1.0, 10.0
    # power=2.0 (aggressive taper)
    strategy = TimestampThickness(min_thickness=min_t, max_thickness=max_t, power=2.0)
    thickness = strategy.compute_thickness(simple_network)

    assert thickness.min() == pytest.approx(min_t)
    assert thickness.max() == pytest.approx(max_t)
    # Midpoint (t=1.5) would be 0.5 normalized. 0.5^2 = 0.25.
    # Node 1 (t=1) is (3-1)/3 = 0.66 normalized. 0.66^2 = 0.44.
    assert thickness[1] < (min_t + 0.66 * (max_t - min_t))

def test_hierarchy_thickness_range(simple_network):
    min_t, max_t = 1.0, 10.0
    strategy = HierarchyThickness(min_thickness=min_t, max_thickness=max_t, power=1.0)
    thickness = strategy.compute_thickness(simple_network)

    assert thickness.min() == pytest.approx(min_t)
    assert thickness.max() == pytest.approx(max_t)
    # Node 0 (depth 0) should be thickest
    assert thickness[0] == max_t
    # Nodes 2, 3 (depth 2) should be thinnest
    assert thickness[2] == min_t
    assert thickness[3] == min_t

def test_descendant_thickness_range(simple_network):
    min_t, max_t = 1.0, 10.0
    # power=0.5 (gentle taper)
    strategy = DescendantThickness(min_thickness=min_t, max_thickness=max_t, power=0.5, mode="leaves_only")
    thickness = strategy.compute_thickness(simple_network)

    # Counts: Node 0: 2 leaves, Node 1: 2 leaves, Node 2: 1 leaf, Node 3: 1 leaf
    # Normalized: 0, 1: 1.0, 2, 3: 0.0
    # Wait, if counts are [2, 2, 1, 1], min=1, max=2.
    # Norm = (count - 1) / (2 - 1) -> [1, 1, 0, 0]
    assert thickness.min() == pytest.approx(min_t)
    assert thickness.max() == pytest.approx(max_t)
    assert thickness[0] == max_t
    assert thickness[2] == min_t

def test_descendant_thickness_power_range(simple_network):
    # Test that even with extreme power, range is respected
    min_t, max_t = 1.0, 10.0
    strategy = DescendantThickness(min_thickness=min_t, max_thickness=max_t, power=0.1, mode="leaves_only")
    thickness = strategy.compute_thickness(simple_network)

    assert thickness.min() == pytest.approx(min_t)
    assert thickness.max() == pytest.approx(max_t)

def test_extra_kwargs_ignored():
    # Should not crash when passed unknown kwargs
    TimestampThickness(min_thickness=1, max_thickness=5, power=1, unknown_param="value")
    HierarchyThickness(min_thickness=1, max_thickness=5, power=1, unknown_param="value")
    DescendantThickness(min_thickness=1, max_thickness=5, power=1, mode="all_nodes", unknown_param="value")
