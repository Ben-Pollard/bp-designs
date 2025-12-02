#!/usr/bin/env python3
"""Smoke test for basic pattern generation after refactor."""

import numpy as np

from bp_designs.core.combinator import PatternCombinator
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.cellular.voronoi import VoronoiTessellation


def test_basic_tree():
    """Can we generate a basic tree?"""
    print("=== Testing Space Colonization ===")
    gen = SpaceColonization(width=100, height=100, num_attractions=100, seed=42)
    tree = gen.generate_pattern()
    print(f"Generated tree with {len(tree.positions)} nodes")

    # Can we query it as a field?
    points = np.array([[50, 50]])
    distance = tree.sample_field(points, "distance")
    print(f"Distance at (50,50): {distance}")

    # Can we render it?
    geometry = tree.to_geometry()
    print(f"Geometry has {len(geometry)} polylines")
    return tree


def test_basic_voronoi():
    """Can we generate a basic Voronoi pattern?"""
    print("\n=== Testing Voronoi Tessellation ===")
    gen = VoronoiTessellation(seed=42, num_sites=20)
    pattern = gen.generate_pattern()
    print(f"Generated Voronoi with {len(pattern.sites)} sites")

    points = np.array([[50, 50]])
    cell_id = pattern.sample_field(points, "cell_id")
    print(f"Cell ID at (50,50): {cell_id}")

    geometry = pattern.to_geometry()
    print(f"Geometry has {len(geometry)} polylines")
    return pattern


def test_composition():
    """Test basic composition operations."""
    print("\n=== Testing Composition ===")
    # Generate patterns
    voronoi = VoronoiTessellation(seed=42, num_sites=30).generate_pattern()
    tree_gen = SpaceColonization(width=100, height=100, num_attractions=200, seed=123)

    # Test guide combinator
    print("Testing PatternCombinator.guide()...")
    guided_tree = PatternCombinator.guide(
        structure=voronoi,
        generator=tree_gen,
        influence_channel="boundary_distance",
        influence_strength=0.7,
    )
    print(f"Guided tree generated: {len(guided_tree.positions)} nodes")

    # Test texture combinator
    print("Testing PatternCombinator.texture()...")
    tree = tree_gen.generate_pattern()
    textured = PatternCombinator.texture(skeleton=tree, fill=voronoi, distance_threshold=5.0)
    print("Textured pattern created (composite)")

    # Test blend combinator
    print("Testing PatternCombinator.blend()...")
    blended = PatternCombinator.blend(
        pattern_a=tree,
        pattern_b=voronoi,
        blend_mode="multiply",
        channel_a="density",
        channel_b="boundary_distance",
    )
    print("Blended pattern created (composite)")

    return guided_tree, textured, blended


def main():
    """Run all smoke tests."""
    print("Running smoke tests for pattern generation system...")

    try:
        test_basic_tree()
    except Exception as e:
        print(f"FAILED basic tree generation: {e}")
        return 1

    try:
        test_basic_voronoi()
    except Exception as e:
        print(f"FAILED basic Voronoi generation: {e}")
        return 1

    try:
        guided, textured, blended = test_composition()
    except Exception as e:
        print(f"FAILED composition tests: {e}")
        return 1

    print("\n✅ All smoke tests passed!")
    return 0


if __name__ == "__main__":
    exit(main())
