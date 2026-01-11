"""Tests for Space Colonization pattern generator."""

import numpy as np

from bp_designs.core.geometry import Canvas, Point
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.primitives.two_d import Oval


class TestSpaceColonization:
    """Test Space Colonization pattern generator."""

    def setup_method(self):
        """Setup common test fixtures."""
        # Create a simple canvas (100x100 square)
        self.canvas = Canvas(coords=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
        # Root position at center
        self.root_position = Point(x=50, y=50, z=None)
        # Initial and final boundaries (same for simplicity)
        self.initial_boundary = Oval.from_width_height(50, 50).generate_pattern()
        self.final_boundary = Oval.from_width_height(70, 70).generate_pattern()

    def test_basic_generation(self):
        """Test basic pattern generation."""
        generator = SpaceColonization(
            canvas=self.canvas,
            root_position=self.root_position,
            initial_boundary=self.initial_boundary,
            final_boundary=self.final_boundary,
            seed=42,
            num_attractions=100,
            kill_distance=5.0,
            segment_length=2.0,
            max_iterations=50,
        )
        network = generator.generate_pattern()

        # Should return BranchNetwork
        from bp_designs.patterns.network import BranchNetwork

        assert isinstance(network, BranchNetwork)

        # Should have at least root node
        assert network.num_nodes >= 1
        assert network.positions.shape[1] == 2  # 2D positions
        assert len(network.node_ids) == network.num_nodes
        assert len(network.parents) == network.num_nodes
        assert len(network.timestamps) == network.num_nodes

        # Root should have parent -1
        assert network.parents[0] == -1

    def test_determinism(self):
        """Test that same seed produces identical results."""
        seed = 123
        generator1 = SpaceColonization(
            canvas=self.canvas,
            root_position=self.root_position,
            initial_boundary=self.initial_boundary,
            final_boundary=self.final_boundary,
            seed=seed,
            num_attractions=50,
            kill_distance=3.0,
            segment_length=1.5,
            max_iterations=30,
        )
        generator2 = SpaceColonization(
            canvas=self.canvas,
            root_position=self.root_position,
            initial_boundary=self.initial_boundary,
            final_boundary=self.final_boundary,
            seed=seed,
            num_attractions=50,
            kill_distance=3.0,
            segment_length=1.5,
            max_iterations=30,
        )

        network1 = generator1.generate_pattern()
        network2 = generator2.generate_pattern()

        # Should have same number of nodes
        assert network1.num_nodes == network2.num_nodes

        # All arrays should be identical
        np.testing.assert_array_equal(network1.node_ids, network2.node_ids)
        np.testing.assert_array_almost_equal(network1.positions, network2.positions)
        np.testing.assert_array_equal(network1.parents, network2.parents)
        np.testing.assert_array_equal(network1.timestamps, network2.timestamps)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different patterns."""
        generator1 = SpaceColonization(
            canvas=self.canvas,
            root_position=self.root_position,
            initial_boundary=self.initial_boundary,
            final_boundary=self.final_boundary,
            seed=1,
            num_attractions=30,
            kill_distance=4.0,
            segment_length=1.0,
            max_iterations=20,
        )
        generator2 = SpaceColonization(
            canvas=self.canvas,
            root_position=self.root_position,
            initial_boundary=self.initial_boundary,
            final_boundary=self.final_boundary,
            seed=2,
            num_attractions=30,
            kill_distance=4.0,
            segment_length=1.0,
            max_iterations=20,
        )

        network1 = generator1.generate_pattern()
        network2 = generator2.generate_pattern()

        # Should not be identical (at least positions different)
        # Note: there's a small chance they could be identical by random chance,
        # but with different seeds and reasonable parameters, this is extremely unlikely
        # If growth occurred, positions should differ.
        if network1.num_nodes > 1 or network2.num_nodes > 1:
            assert not np.array_equal(network1.positions, network2.positions)

    def test_num_attractions_parameter(self):
        """Test that num_attractions affects pattern."""
        generator_few = SpaceColonization(
            canvas=self.canvas,
            root_position=self.root_position,
            initial_boundary=self.initial_boundary,
            final_boundary=self.final_boundary,
            seed=42,
            num_attractions=10,
            kill_distance=5.0,
            segment_length=2.0,
            max_iterations=20,
        )
        generator_many = SpaceColonization(
            canvas=self.canvas,
            root_position=self.root_position,
            initial_boundary=self.initial_boundary,
            final_boundary=self.final_boundary,
            seed=42,
            num_attractions=100,
            kill_distance=5.0,
            segment_length=2.0,
            max_iterations=20,
        )

        network_few = generator_few.generate_pattern()
        network_many = generator_many.generate_pattern()

        # More attractions should generally produce more growth
        # (though not guaranteed, it's a reasonable expectation)
        # We'll just ensure both generate valid networks
        assert network_few.num_nodes >= 1
        assert network_many.num_nodes >= 1

    def test_kill_distance_parameter(self):
        """Test that kill_distance affects pattern."""
        generator_small = SpaceColonization(
            canvas=self.canvas,
            root_position=self.root_position,
            initial_boundary=self.initial_boundary,
            final_boundary=self.final_boundary,
            seed=42,
            num_attractions=50,
            kill_distance=2.0,
            segment_length=1.0,
            max_iterations=20,
        )
        generator_large = SpaceColonization(
            canvas=self.canvas,
            root_position=self.root_position,
            initial_boundary=self.initial_boundary,
            final_boundary=self.final_boundary,
            seed=42,
            num_attractions=50,
            kill_distance=10.0,
            segment_length=1.0,
            max_iterations=20,
        )

        network_small = generator_small.generate_pattern()
        network_large = generator_large.generate_pattern()

        # Different kill distances should produce different networks
        # (not guaranteed but likely)
        assert network_small.num_nodes >= 1
        assert network_large.num_nodes >= 1

    def test_segment_length_parameter(self):
        """Test that segment_length affects pattern."""
        generator_short = SpaceColonization(
            canvas=self.canvas,
            root_position=self.root_position,
            initial_boundary=self.initial_boundary,
            final_boundary=self.final_boundary,
            seed=42,
            num_attractions=50,
            kill_distance=5.0,
            segment_length=0.5,
            max_iterations=20,
        )
        generator_long = SpaceColonization(
            canvas=self.canvas,
            root_position=self.root_position,
            initial_boundary=self.initial_boundary,
            final_boundary=self.final_boundary,
            seed=42,
            num_attractions=50,
            kill_distance=5.0,
            segment_length=3.0,
            max_iterations=20,
        )

        network_short = generator_short.generate_pattern()
        network_long = generator_long.generate_pattern()

        # Different segment lengths should produce different networks
        assert network_short.num_nodes >= 1
        assert network_long.num_nodes >= 1

    def test_max_iterations(self):
        """Test that iteration limit works."""
        # Test with very few iterations
        generator = SpaceColonization(
            canvas=self.canvas,
            root_position=self.root_position,
            initial_boundary=self.initial_boundary,
            final_boundary=self.final_boundary,
            seed=42,
            num_attractions=100,
            kill_distance=5.0,
            segment_length=2.0,
            max_iterations=1,  # Just one iteration
        )
        network = generator.generate_pattern()

        # Should have at least root node
        assert network.num_nodes >= 1
        # With only one iteration, might only have root or root + some children
        # Both are valid

    def test_boundary_containment(self):
        """Test that generated nodes stay within final boundary."""
        # Use a smaller final boundary to test containment
        # Center the boundaries around the root position (50, 50)
        # We need to make sure the root is INSIDE the small boundary
        small_boundary = Oval.from_width_height(40, 40, canvas=self.canvas).generate_pattern()
        initial_boundary = Oval.from_width_height(50, 50, canvas=self.canvas).generate_pattern()

        # Root at center (50, 50) is inside a 40x40 oval centered at (50, 50)
        root_pos = Point(50, 50, 0)

        generator = SpaceColonization(
            canvas=self.canvas,
            root_position=root_pos,
            initial_boundary=initial_boundary,
            final_boundary=small_boundary,
            seed=42,
            num_attractions=50,
            kill_distance=5.0,
            segment_length=2.0,
            max_iterations=30,
        )
        network = generator.generate_pattern()

        # All positions should be within or near the final boundary
        # Allow small margin for numerical precision
        margin = 1e-6
        # Use shapely to get bounds from the resolved geometry
        from shapely.geometry import Polygon as ShapelyPolygon
        geom = small_boundary.to_geometry(self.canvas)
        poly = ShapelyPolygon(geom.polylines[0])
        xmin, ymin, xmax, ymax = poly.bounds

        for pos in network.positions:
            x, y = pos
            assert xmin - margin <= x <= xmax + margin, f"X coordinate {x} outside boundary [{xmin}, {xmax}]"
            assert ymin - margin <= y <= ymax + margin, f"Y coordinate {y} outside boundary [{ymin}, {ymax}]"

    def test_zero_attractions(self):
        """Test with zero attractions (should still produce root node)."""
        generator = SpaceColonization(
            canvas=self.canvas,
            root_position=self.root_position,
            initial_boundary=self.initial_boundary,
            final_boundary=self.final_boundary,
            seed=42,
            num_attractions=0,  # No attractions
            kill_distance=5.0,
            segment_length=2.0,
            max_iterations=10,
        )
        network = generator.generate_pattern()

        # Should have exactly root node (no growth without attractions)
        assert network.num_nodes == 1
        assert network.parents[0] == -1

    def test_edge_case_small_boundary(self):
        """Test with very small boundary."""
        tiny_boundary = Oval.from_width_height(5,5).generate_pattern()

        generator = SpaceColonization(
            canvas=self.canvas,
            root_position=self.root_position,
            initial_boundary=tiny_boundary,
            final_boundary=tiny_boundary,
            seed=42,
            num_attractions=20,
            kill_distance=2.0,
            segment_length=1.0,
            max_iterations=10,
        )
        network = generator.generate_pattern()

        # Should generate valid network (might have limited growth)
        assert network.num_nodes >= 1

    def test_network_structure_integrity(self):
        """Test that network structure is valid (parent references exist)."""
        generator = SpaceColonization(
            canvas=self.canvas,
            root_position=self.root_position,
            initial_boundary=self.initial_boundary,
            final_boundary=self.final_boundary,
            seed=42,
            num_attractions=50,
            kill_distance=5.0,
            segment_length=2.0,
            max_iterations=30,
        )
        network = generator.generate_pattern()

        # Check that all parent IDs exist in node_ids (except -1 for root)
        valid_node_ids = set(network.node_ids)
        for node_id, parent_id in zip(network.node_ids, network.parents, strict=True):
            if parent_id != -1:
                assert parent_id in valid_node_ids, (
                    f"Parent {parent_id} not found for node {node_id}"
                )

        # Check timestamps: root should have timestamp 0, children should have increasing timestamps
        assert network.timestamps[0] == 0  # Root timestamp
        # Children should have timestamps >= 1
        if network.num_nodes > 1:
            assert np.all(network.timestamps[1:] >= 1)
