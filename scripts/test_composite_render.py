"""Test composite pattern rendering."""

import sys

sys.path.insert(0, "src")

from bp_designs.core.combinator import PatternCombinator
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.cellular.voronoi import VoronoiTessellation

# Generate simple patterns
tree_gen = SpaceColonization(seed=42, width=100, height=100, num_attractions=100)
voronoi_gen = VoronoiTessellation(seed=42, num_sites=20, relaxation_iterations=2)

tree = tree_gen.generate_pattern()
voronoi = voronoi_gen.generate_pattern()

print("Testing texture composition...")
textured = PatternCombinator.texture(skeleton=tree, fill=voronoi, distance_threshold=8.0)
print(f"Textured type: {type(textured)}")
print(f"Metadata: {textured.metadata}")
try:
    geometry = textured.to_geometry()
    print(f"Geometry generated: {len(geometry)} polylines")
except Exception as e:
    import traceback

    print(f"Error: {e}")
    traceback.print_exc()

print("\nTesting blend composition...")
blended = PatternCombinator.blend(
    pattern_a=tree,
    pattern_b=voronoi,
    blend_mode="multiply",
    channel_a="density",
    channel_b="boundary_distance",
)
print(f"Blended type: {type(blended)}")
print(f"Metadata: {blended.metadata}")
try:
    geometry = blended.to_geometry()
    print(f"Geometry generated: {len(geometry)} polylines")
except Exception as e:
    import traceback

    print(f"Error: {e}")
    traceback.print_exc()
