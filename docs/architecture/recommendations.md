# Architectural Recommendations - BP Designs

This document provides a comprehensive set of recommendations for the `bp-designs` project, comparing the current implementation against industry standards and best practices.

## 1. Scalability
*How well does the system handle larger patterns, more complex scenes, and massive experiment batches?*

### Current State
- Uses `scipy.spatial.cKDTree` for efficient spatial queries in `SpaceColonization`.
- `ExperimentRunner` utilizes `ProcessPoolExecutor` for parallel variant generation.
- `BranchNetwork` uses `numpy` for data storage.

### Recommendations
- **Vectorize Semantic Queries**: Current implementations of `depths` and `branch_ids` in [`src/bp_designs/patterns/network/base.py`](src/bp_designs/patterns/network/base.py) use Python loops. These should be vectorized using `numpy` or graph libraries (like `networkx` or `igraph`) to handle networks with millions of nodes.
- **Efficient Batch Storage**: The `ExperimentRunner` currently saves each variant as individual SVG and JSON files. For massive batches (10,000+ variants), this can lead to filesystem performance issues. Consider:
    - Using a single database (e.g., SQLite) or a structured binary format (e.g., HDF5, Zarr) for metadata and raw pattern data.
    - Implementing a "lazy rendering" approach where SVGs are only generated on demand in the gallery.
- **Level of Detail (LOD)**: Implement LOD strategies in the rendering pipeline. For very large patterns, render simplified versions (e.g., fewer segments, no organs) when viewed at a distance or for quick previews.
- **Memory Management**: Ensure that large `numpy` arrays are handled efficiently, using `memmap` if necessary for extremely large datasets that exceed RAM.

## 2. Maintainability
*Assess the use of SOLID principles, Clean Architecture, and the clarity of the three-layer system.*

### Current State
- Strong three-layer abstraction: Generator → Pattern → Geometry.
- Strategy pattern used for growth, attraction, and topology.
- Pydantic used for `RenderStyle` validation.

### Recommendations
- **Single Responsibility Principle (SRP)**: The `BranchNetwork` class in [`src/bp_designs/patterns/network/base.py`](src/bp_designs/patterns/network/base.py) is currently a "God Object" handling data, validation, semantic queries, refinement logic, and rendering delegation.
    - **Action**: Move refinement logic (`relocate`, `subdivide`, `decimate`) to separate `RefinementStrategy` classes.
    - **Action**: Move semantic query logic to a `NetworkAnalyzer` or similar utility.
- **Interface Segregation**: Define explicit interfaces (Protocols or ABCs) for all strategies (Growth, Attraction, Topology, Thickness, Color) to ensure they remain decoupled and easily swappable.
- **Dependency Inversion**: Ensure that high-level modules (Generators) do not depend on low-level implementation details of Patterns. The current use of `to_geometry` as an interchange layer is a good start, but ensure that `Pattern` implementations don't leak algorithm-specific details.
- **Refactor Renderer**: The `NetworkRenderer.render` method in [`src/bp_designs/patterns/network/renderer.py`](src/bp_designs/patterns/network/renderer.py) is becoming monolithic. Break it down into smaller, focused methods for each render mode (polyline, polygon, smooth, blocky).

## 3. Resilience
*Error handling in generators, validation of parameters, and robustness of the rendering pipeline.*

### Current State
- Basic `ValueError` checks in constructors.
- `ExperimentRunner` catches exceptions per variant to prevent batch failure.
- Pydantic used for rendering parameters.

### Recommendations
- **Centralized Parameter Validation**: Use Pydantic models for *all* generator parameters, not just rendering. This provides automatic type checking, range validation, and clear error messages.
- **Graceful Degradation**: Implement fallbacks for missing dependencies (e.g., if `scipy` is missing, use a slower but functional `numpy` implementation, but log a warning).
- **Robust Geometry Processing**: Geometric operations (like `shapely.union_all`) can occasionally fail on degenerate geometry. Wrap these in try-except blocks with sensible fallbacks (e.g., returning the original segments if unioning fails).
- **Input Sanitization**: Validate that input patterns (like boundaries) are topologically sound (e.g., closed polygons, no self-intersections) before passing them to generators.

## 4. Craft & Manufacturing Standards
*Alignment with CAD/CAM standards (SVG, DXF, STL) and physical constraint validation.*

### Current State
- SVG is the primary output format.
- `to_polygon` creates unioned "skins" suitable for CNC/Laser cutting.
- Tapering and thickness strategies consider physical constraints.

### Recommendations
- **Expanded Export Formats**: Add explicit support for DXF (using `ezdxf`) and STL (using `numpy-stl` or `trimesh`). While SVG is good for 2D, DXF is the industry standard for CNC, and STL is required for 3D-printed stamps.
- **Manufacturing Constraint Validation**: Implement a "Pre-flight Check" system that validates patterns against physical constraints *before* export:
    - **Minimum Feature Size**: Check if any part of the pattern is thinner than the manufacturing limit (e.g., 0.5mm).
    - **Minimum Spacing**: Check if gaps between strokes are too small, which could lead to "detail collapse" during embossing.
    - **Closed Paths**: Ensure all polygons intended for cutting are closed and manifold.
- **Unit Awareness**: Ensure all coordinates and dimensions are explicitly tied to physical units (mm, inches) throughout the pipeline, rather than just "canvas units".

## 5. Testing & Quality
*Best practices for testing generative systems (visual regression, property-based testing).*

### Current State
- Standard unit tests for core logic and determinism.
- Determinism tests ensure same seed → same output.

### Recommendations
- **Visual Regression Testing**: Generative systems are hard to test with unit tests alone. Implement visual regression tests that render patterns to SVGs/PNGs and compare them against "golden" baselines using tools like `pytest-regressions` or custom image comparison.
- **Property-Based Testing**: Use `Hypothesis` to test generators with a wide range of random parameters. This can uncover edge cases where algorithms fail (e.g., zero attractions, extremely small boundaries, degenerate input geometry).
- **Topology Validation**: Add tests that specifically check the topological integrity of generated networks (e.g., no cycles, all nodes connected to a root, no overlapping nodes).
- **Performance Benchmarking**: Implement automated benchmarks (using `pytest-benchmark`) for core algorithms to detect performance regressions as the system evolves.

## 6. Twelve-Factor App Principles (Applicability)
*While not a web app, some principles apply to the experimentation system.*

- **Config**: Keep all experiment configurations in environment variables or external files (JSON/YAML), never hardcoded.
- **Backing Services**: Treat the gallery and any future vector databases as attached resources.
- **Build, Release, Run**: Maintain a strict separation between the code (build), the experiment configuration (release), and the execution (run).
- **Concurrency**: The `ExperimentRunner` already follows the concurrency principle by scaling out via processes.
- **Logs**: Implement structured logging (e.g., using `structlog`) to make it easier to debug failed experiment variants in large batches.

## 7. Design Patterns
- **Strategy Pattern**: Continue using and formalizing the Strategy pattern for all swappable logic.
- **Factory Pattern**: Consider a `GeneratorFactory` to instantiate generators from configuration strings/dicts, especially useful for the `ExperimentRunner`.
- **Decorator Pattern**: Could be used for post-processing patterns (e.g., a `SmoothDecorator` that wraps a `Pattern` and applies smoothing to its geometry).
- **Observer Pattern**: Useful if the `ExperimentRunner` needs to provide real-time updates to a UI or dashboard.
