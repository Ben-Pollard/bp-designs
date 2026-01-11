# Organic Rendering: Outline-based Skinning

## Overview
To move beyond simple line segments and achieve truly organic, nature-inspired forms, we propose implementing an "Outline-based Skinning" algorithm. This approach treats the branching network as a skeleton and generates a single continuous polygon "skin" around it.

## Core Algorithm: The "Envelope" Method
The most robust way to generate a smooth outline for a tree structure is to union the envelopes of its segments.

1.  **Node Radii**: For every node $i$, define a radius $R_i$.
    -   **Leonardo's Rule**: Use the pipe model ($R_{parent}^n = \sum R_{child\_i}^n$) where $n \approx 2$ to ensure natural-looking thickness transitions at branching points.
2.  **Segment Envelopes**: For every segment connecting parent $P$ and child $C$:
    -   Create a tapered capsule (a trapezoid defined by the external tangents of the circles at $P$ and $C$).
    -   This ensures the "skin" is perfectly flush with the nodes.
3.  **Global Union**: Perform a boolean union of all segment envelopes using a library like Shapely.

## Advantages
-   **Manufacturing Fidelity**: Produces a single closed path (Fill) rather than a set of overlapping strokes. This is critical for laser cutting and embossing.
-   **Natural Branching**: The union operation automatically handles complex joints and branching points, creating smooth transitions.
-   **Manufacturing Ready**: Produces a single clean vector path, ideal for laser cutting, embossing, or CNC milling.
-   **Variable Tapering**: Supports smooth tapering along the entire length of a branch.
-   **No "Thick Neck"**: Fundamentally solves the issue of side branches appearing too thick at their origin.

## Implementation Strategy

### Phase 1: Prototype with Shapely
-   Implement `BranchNetwork.to_polygon()` using `shapely.ops.unary_union`.
-   For each segment $P \to C$, calculate the trapezoid formed by the circles $C_P(R_P)$ and $C_C(R_C)$.
-   Use `shapely.geometry.Polygon` for each segment and union them.

### Phase 2: Tapered Envelopes
-   To handle tapering within a single segment, we can calculate the four vertices of the trapezoid that forms the envelope between two circles of different radii.
-   Vertices are found by calculating the tangent points between the two circles.

### Phase 3: Optimization
-   Unioning hundreds of polygons can be computationally expensive.
-   **Optimization 1**: Use a spatial index (like STRtree) to only union overlapping segments.
-   **Optimization 2**: Union branches into sub-polygons first, then union the sub-polygons.

## API Integration
-   Add `to_polygon()` to the `Pattern` or `BranchNetwork` interface.
-   Update `to_svg()` to support a `render_style='organic'` option that utilizes the polygon representation.

## References
-   *The Algorithmic Beauty of Plants* (Prusinkiewicz & Lindenmayer)
-   *Space Colonization Algorithm for Leaf Venation* (Runions et al.)
