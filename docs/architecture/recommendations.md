# Flow Fields — Architectural Recommendations

This document compares the proposed Flow Field implementation (from [`docs/design/flow_fields.md`](../design/flow_fields.md)) against the existing architecture and industry best practices for generative systems.

## 1. Architectural Alignment

The proposed design for Flow Fields aligns well with the existing **Three-Layer System** (Algorithm, Semantic, Interchange):

- **Algorithm Layer:** The "Field Definition" (Trigonometric, Radial, Noise) maps to the `Generator` concept.
- **Semantic Layer:** The "Streamline" maps to a new `StreamlinePattern` (or a more general `CurveNetworkPattern`).
- **Interchange Layer:** The resulting polylines map to the existing `Polyline` and `PointSet` geometry.

### 1.1 Strategy Pattern Consistency
The use of "Strategies" for Length, Width, Density, and Seeding is highly consistent with the `SpaceColonization` implementation in [`src/bp_designs/generators/branching/strategies.py`](../../src/bp_designs/generators/branching/strategies.py). 

**Recommendation:** Formalize these as ABCs within a new `bp_designs.generators.flow` module to maintain the "swappable behavior" pattern.

## 2. Best Practice Comparison

### 2.1 Functional Composition vs. Object-Oriented State
The design document suggests fields are "mathematical functions evaluated at any point." 

- **Standard Practice:** Many flow field implementations use a discrete grid (vector field).
- **Our Design:** Analytical functions (resolution-independent) are superior for our "Craft Realism" goal as they avoid aliasing and support infinite zoom/detail.
- **Recommendation:** Implement fields as `Callable[[np.ndarray], np.ndarray]` (mapping `(N, 2)` positions to `(N, 2)` vectors) to allow for easy composition via functional wrapping (e.g., `field = lambda p: field1(p) + field2(p)`).

### 2.2 Decoupling Field from Integration
The document correctly separates "Field Definition" from "Field Visualization."

- **Recommendation:** The `FlowField` should be a first-class object (likely a `Pattern` or a new `Field` type) that can be passed to *other* generators. For example, `SpaceColonization` could use a `FlowField` as a `direction_guide`.

### 2.3 Integration Methods
Streamline generation requires numerical integration (Euler, Runge-Kutta).

- **Recommendation:** Use RK4 (Runge-Kutta 4th Order) as the default integration method. While Euler is simpler, RK4 is significantly more stable for the "turbulent" fields described in the design doc, preventing streamlines from "drifting" off the mathematical path.

## 3. Categorized Recommendations

### 3.1 Scalability
- **Spatial Lookups:** For "Termination" strategies that stop when approaching existing streamlines, a KDTree (from `scipy.spatial`) is essential. This is already identified as a need in [`docs/architecture/plan_implications.md`](plan_implications.md).
- **Vectorization:** Ensure the integration loop is vectorized where possible (integrating multiple streamlines in parallel).

### 3.2 Security & Resilience
- **Determinism:** Ensure all noise-derived fields use the project's `seed` mechanism. Avoid global `np.random` calls.
- **Boundary Safety:** Implement robust "Termination" at canvas boundaries to prevent infinite loops or memory exhaustion in integration.

### 3.3 Maintainability
- **Parameter Passing:** The user requested "parameters passed more deliberately via strategies." 
- **Recommendation:** Use Pydantic models for Strategy configurations (similar to `NetworkStyle`) to ensure type safety and clear documentation of what each strategy requires.

### 3.4 Craft-Awareness
- **Tapering:** The "Width" strategy should support the same `taper_power` and `taper_style` logic used in `BranchNetwork` to ensure visual consistency across the library.
- **SVG Optimization:** Long streamlines can result in massive SVG files. Implement a "Simplification" step (e.g., Ramer-Douglas-Peucker) as part of the `StreamlinePattern` to reduce point count without losing visual character.
