"""Tests for Voronoi tessellation pattern generator."""

import numpy as np
import pytest

from bp_designs.core.geometry import Geometry
from bp_designs.generators.cellular.voronoi import VoronoiTessellation


class TestVoronoiTessellation:
    """Test Voronoi pattern generator."""

    def test_basic_generation(self):
        """Test basic pattern generation."""
        generator = VoronoiTessellation(seed=42, num_sites=10)
        geometry = generator.generate_pattern().to_geometry()

        # Should return Geometry instance
        assert isinstance(geometry, Geometry)
        assert len(geometry.polylines) > 0

        # Each polyline should be numpy array with shape (N, 2)
        for polyline in geometry.polylines:
            assert isinstance(polyline, np.ndarray)
            assert polyline.ndim == 2
            assert polyline.shape[1] == 2

    def test_determinism(self):
        """Test that same seed produces identical results."""
        seed = 123
        params = {
            "num_sites": 20,
            "relaxation_iterations": 2,
            "render_mode": "edges",
            "width": 100.0,
            "height": 100.0,
        }

        gen1 = VoronoiTessellation(seed=seed, **params)
        gen2 = VoronoiTessellation(seed=seed, **params)

        geom1 = gen1.generate_pattern().to_geometry()
        geom2 = gen2.generate_pattern().to_geometry()

        # Should have same number of polylines
        assert len(geom1.polylines) == len(geom2.polylines)

        # Each polyline should be identical
        for p1, p2 in zip(geom1.polylines, geom2.polylines, strict=True):
            np.testing.assert_array_almost_equal(p1, p2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different patterns."""
        gen1 = VoronoiTessellation(seed=1, num_sites=20)
        gen2 = VoronoiTessellation(seed=2, num_sites=20)

        geom1 = gen1.generate_pattern().to_geometry()
        geom2 = gen2.generate_pattern().to_geometry()

        # Should not be identical (at least some polylines different)
        all_same = True
        for p1, p2 in zip(geom1.polylines, geom2.polylines, strict=False):
            if not np.allclose(p1, p2):
                all_same = False
                break

        assert not all_same, "Different seeds should produce different patterns"

    def test_render_mode_edges(self):
        """Test edges render mode."""
        generator = VoronoiTessellation(seed=42, num_sites=15, render_mode="edges")
        geometry = generator.generate_pattern().to_geometry()

        assert len(geometry.polylines) > 0
        # Edges should be 2-point lines
        for edge in geometry.polylines:
            assert len(edge) == 2

    def test_render_mode_cells(self):
        """Test cells render mode."""
        generator = VoronoiTessellation(seed=42, num_sites=15, render_mode="cells")
        geometry = generator.generate_pattern().to_geometry()

        assert len(geometry.polylines) > 0
        # Cells should be closed polygons (>= 3 vertices)
        for cell in geometry.polylines:
            assert len(cell) >= 3

    def test_render_mode_both(self):
        """Test both render mode includes edges and cells."""
        generator = VoronoiTessellation(seed=42, num_sites=15, render_mode="both")
        geometry = generator.generate_pattern().to_geometry()

        # Should have both edges (2 points) and cells (3+ points)
        has_edges = any(len(p) == 2 for p in geometry.polylines)
        has_cells = any(len(p) >= 3 for p in geometry.polylines)

        assert has_edges and has_cells

    def test_invalid_render_mode(self):
        """Test that invalid render mode raises ValueError."""
        with pytest.raises(ValueError, match="render_mode must be"):
            VoronoiTessellation(seed=42, render_mode="invalid")

    def test_relaxation_iterations(self):
        """Test that relaxation iterations affect output."""
        gen_no_relax = VoronoiTessellation(seed=42, num_sites=20, relaxation_iterations=0)
        gen_with_relax = VoronoiTessellation(seed=42, num_sites=20, relaxation_iterations=3)

        geom_no_relax = gen_no_relax.generate_pattern().to_geometry()
        geom_with_relax = gen_with_relax.generate_pattern().to_geometry()

        # Relaxation should change the pattern
        all_same = len(geom_no_relax.polylines) == len(geom_with_relax.polylines)
        if all_same:
            for p1, p2 in zip(geom_no_relax.polylines, geom_with_relax.polylines, strict=True):
                if not np.allclose(p1, p2):
                    all_same = False
                    break

        assert not all_same, "Relaxation should change the pattern"

    def test_num_sites_parameter(self):
        """Test that num_sites affects pattern complexity."""
        gen_few = VoronoiTessellation(seed=42, num_sites=5, render_mode="cells")
        gen_many = VoronoiTessellation(seed=42, num_sites=50, render_mode="cells")

        geom_few = gen_few.generate_pattern().to_geometry()
        geom_many = gen_many.generate_pattern().to_geometry()

        # More sites should generally produce more cells
        # (accounting for clipping, may not be exact)
        assert len(geom_many.polylines) > len(geom_few.polylines)

    def test_canvas_bounds(self):
        """Test that all vertices are within or near canvas bounds."""
        width, height = 100.0, 150.0
        generator = VoronoiTessellation(
            seed=42, num_sites=20, width=width, height=height, render_mode="both"
        )
        geometry = generator.generate_pattern().to_geometry()

        # Allow small margin for numerical precision and clipping
        margin = 1.0

        for polyline in geometry.polylines:
            x_coords = polyline[:, 0]
            y_coords = polyline[:, 1]

            assert np.all(x_coords >= -margin), "X coordinates below lower bound"
            assert np.all(x_coords <= width + margin), "X coordinates above upper bound"
            assert np.all(y_coords >= -margin), "Y coordinates below lower bound"
            assert np.all(y_coords <= height + margin), "Y coordinates above upper bound"

    def test_empty_pattern_handling(self):
        """Test handling of edge cases that might produce empty patterns."""
        # Very few sites might still produce some output
        generator = VoronoiTessellation(seed=42, num_sites=2)
        geometry = generator.generate_pattern().to_geometry()

        # Should at least return a Geometry (even if empty in extreme cases)
        assert isinstance(geometry, Geometry)

    def test_large_canvas(self):
        """Test generation on larger canvas."""
        generator = VoronoiTessellation(
            seed=42, num_sites=100, width=500.0, height=500.0, render_mode="edges"
        )
        geometry = generator.generate_pattern().to_geometry()

        assert len(geometry.polylines) > 0
        assert all(isinstance(p, np.ndarray) for p in geometry.polylines)
