# Pattern Composition Architecture Spec

## Context & Motivation

### The Problem
We have multiple pattern generation algorithms (Space Colonization, Voronoi Tessellation, and more to come) that produce different semantic structures:
- `BranchNetwork` - tree-like structures with parents, depths, branch IDs
- `Cells` - cellular patterns with polygons, boundaries, neighbors
- Future: reaction-diffusion, flow fields, L-systems, etc.

Currently, combining these requires writing custom pairwise logic for each combination. With N pattern types, this requires O(N²) implementations - completely unworkable as we add more patterns.

### Design Goals
1. **Composability**: Any two patterns should be combinable without custom glue code
2. **Semantic preservation**: Patterns keep their internal structure (tree semantics, cell semantics, etc.)
3. **Field-based interface**: Patterns expose themselves as queryable spatial fields
4. **Efficient generation**: Use vectorized numpy where possible, Numba where needed
5. **Iterative exploration**: Support rapid visual experimentation with combinations

### Key Insight
All patterns can expose a **field interface** - the ability to sample properties at arbitrary spatial positions. This provides a universal language for composition while preserving each pattern's internal semantic structure.


## Core Interfaces
Core interfaces for the pattern generation system.

Architecture:
    Generator → Pattern → Geometry

    Generator: Algorithm that creates patterns (SpaceColonization, VoronoiTessellation)
    Pattern: Interface for spatial queries + rendering (field interface)
    Geometry: Pure geometric data (list of polylines as ndarrays)

Example:
    generator = SpaceColonization(...)
    pattern = generator.generate_pattern()  # Returns Pattern (e.g., BranchNetwork)
    geometry = pattern.to_geometry()         # Returns Geometry
    svg = geometry_to_svg(geometry)          # Render

## Folder Structure
- Interfaces are defined as ABCs in core.
- generators are in generators, organised into subfolders by the type of pattern they generate
- patterns are in patterns

# ============================================================================
# USAGE EXAMPLES - Showing the Pattern Algebra
# ============================================================================

def example_usage():
    """Examples showing the pattern algebra in action."""

    # 1. Basic unguided generation
    tree_gen = SpaceColonization(bounds=(0, 0, 100, 100), n_attractions=500, seed=42)
    tree = tree_gen.generate_pattern()

    # 2. Query tree as field
    points = np.array([[50, 50], [75, 75]])
    densities = tree.sample_field(points, 'density')
    directions = tree.sample_field(points, 'direction')  # (2, 2) vectors

    # 3. ALGEBRA: Voronoi guides tree growth
    voronoi = VoronoiTessellation(bounds=(0, 0, 100, 100), seed=42).generate_pattern()
    guided_tree = PatternCombinator.guide(
        structure=voronoi,
        generator=tree_gen,
        influence_channel='boundary_distance',
        influence_strength=0.7
    )

    # 4. ALGEBRA: Texture tree with cells
    textured_tree = PatternCombinator.texture(
        skeleton=tree,
        fill=voronoi,
        distance_threshold=5.0
    )

    # 5. ALGEBRA: Blend two patterns
    blended = PatternCombinator.blend(
        pattern_a=tree,
        pattern_b=voronoi,
        blend_mode='multiply',
        channel_a='density',
        channel_b='boundary_distance'
    )

    # 6. RECURSIVE COMPOSITION - Results are Patterns!
    # Can combine composite patterns further
    another_voronoi = VoronoiTessellation(bounds=(0, 0, 100, 100), seed=99).generate_pattern()
    recursive = PatternCombinator.texture(
        skeleton=guided_tree,  # This is a Pattern (from guide())
        fill=another_voronoi,
        distance_threshold=3.0
    )

    # 7. All patterns render the same way
    for pattern in [tree, guided_tree, textured_tree, blended, recursive]:
        geometry = pattern.to_geometry()
        # svg = geometry_to_svg(geometry)

    # 8. All patterns can be queried as fields
    for pattern in [tree, guided_tree, textured_tree]:
        channels = pattern.available_channels()
        for channel_name in channels:
            values = pattern.sample_field(points, channel_name)
            print(f"{channel_name}: {values}")


# This demonstrates:
# - Patterns maintain semantic structures (BranchNetwork with parents, depths)
# - Patterns expose field interface (sample_field)
# - PatternCombinator provides semantic algebra (guide, texture, blend, nest)
# - Operations work on any Pattern via field queries
# - Results are composable (CompositePattern is a Pattern)
