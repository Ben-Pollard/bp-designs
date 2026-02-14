# Plan Implications Audit: Technical Debt & Infrastructure

This document analyzes the technical implications of the upcoming plans outlined in `docs/ROADMAP.md` and `docs/exploration/space_col_artistic_directions.md`.

---

## Phase 1: Space Colonization & Composition Refinement

### 1. Technical Debt
- **Spatial Indexing:** Current "Wild" variants (Parasitic Geometry, Neural Mats) require frequent proximity checks. The current naive $O(N^2)$ or simple spatial lookups in `SpaceColonization` will become a bottleneck.
- **Branch Topology:** `Neural Mats` (cross-linking) and `Temporal Echoes` (isochrones) require a true Graph structure. The current `BranchNetwork` is likely optimized for trees (parent-child). Refactoring to a more general `GraphPattern` or `NetworkPattern` is needed.
- **Parameter Bloat:** "Wild" variants introduce many new parameters (momentum, inertia, quantization). The `SpaceColonization` class risks becoming a "God Class."

### 2. Breaking Changes
- **`BranchNetwork` API:** Moving from tree-only to graph-topology (fusion/cross-linking) will break assumptions in rendering and thickness calculation that rely on a single root or acyclic paths.
- **`Generator.generate_pattern` Signature:** Adding `density_guide` and `direction_guide` as standard inputs (as per `docs/api.md`) will require updating all existing generator implementations.

### 3. Infrastructure Requirements
- **Spatial Indexing Library:** Integration of `scipy.spatial.KDTree` or `rtree` for efficient neighbor searches in "Wild" variants.
- **Graph Library:** Potential need for `networkx` for complex topology analysis (fusion, cycle detection in Neural Mats).

### 4. Complexity Risk
- **Determinism in "Wild" Variants:** Momentum and inertia calculations must be carefully implemented to remain deterministic across different platforms/float precisions.
- **Guided Growth:** Vortex fields and noise-driven density introduce external dependencies (noise functions) that must be seeded correctly to maintain reproducibility.

---

## Phase 2 & 3: Voronoi & 3D Transition

### 1. Technical Debt
- **2D/3D Duality:** The current `Geometry` and `Pattern` interfaces are heavily 2D-centric (e.g., `to_svg`). Supporting 3D lighting and geometry will require a fundamental shift in the `Renderer` and `Geometry` ABCs.
- **Voronoi Architecture:** The roadmap notes Voronoi needs to be "brought up to standard." This implies the current implementation is a legacy outlier that doesn't follow the `Generator -> Pattern -> Geometry` flow.

### 2. Breaking Changes
- **`Renderer` Interface:** Transitioning from SVG-only to 3D-capable rendering will likely break the current `to_svg` pattern if not handled via a multi-backend renderer.
- **Coordinate Systems:** Moving to 3D might require changing `tuple[float, float]` to `tuple[float, ...]` or a dedicated `Vector` class across the entire codebase.

### 3. Infrastructure Requirements
- **3D Rendering Engine:** `PyVista`, `Trimesh`, or `Open3D` for 3D visualization and mesh generation.
- **3D Geometry Library:** `Shapely` is 2D; `PyVista` or `GTS` might be needed for 3D boolean operations (Parasitic Geometry in 3D).

### 4. Complexity Risk
- **3D Performance:** Generating complex branching structures in 3D with "fleshy" skinning (Metaballs) is computationally expensive and risks losing the "playful iteration" speed.

---

## Phase 4 - 6: Advanced Composition & Evaluation

### 1. Technical Debt
- **Mutual Influence:** Phase 6 mentions "mutual influence" between patterns. This requires a "Blackboard" or "Scene" architecture where generators can query the state of other patterns, which is not currently supported by the pure `Generator` API.
- **Masking Logic:** Advanced masking requires robust boolean operations on `Geometry` types, which is currently minimal.

### 2. Breaking Changes
- **`Scene` vs `Pattern`:** The introduction of a `Scene` or `Canvas` that manages multiple patterns might change how `ExperimentRunner` interacts with individual generators.

### 3. Infrastructure Requirements
- **Vector Database / Spatial DB:** Phase 8 mentions "Advanced knowledge management." This might require a local vector store (e.g., `ChromaDB` or `DuckDB` with spatial extensions) to catalog and search "Moves."
- **Manufacturing Exports:** Libraries for G-Code, STL, or DXF export (e.g., `ezdxf`, `numpy-stl`).

### 4. Complexity Risk
- **Heuristic Evaluation:** Phase 5 (Pattern Evaluation) introduces subjective "naturalness" scores. If implemented via ML, it adds significant environment complexity (PyTorch/TensorFlow). If heuristic, it risks being arbitrary.

---

## Summary of High-Risk Areas

| Feature | Risk Type | Impact |
|---------|-----------|--------|
| **Graph Fusion** | Breaking Change | High: Affects all downstream rendering/thickness logic. |
| **3D Transition** | Infrastructure | High: Requires new stack (Open3D/Trimesh) and API overhaul. |
| **Mutual Influence** | Architecture | Medium: Requires moving from "Pure" generators to "Context-aware" ones. |
| **Spatial Indexing** | Technical Debt | Medium: Essential for performance as complexity grows. |
