# Future Architectural Direction — Flow Fields

This document defines the agreed architectural direction for the Flow Field implementation, ensuring alignment with existing patterns while improving parameter clarity and composability.

## 1. Core Abstractions

The Flow Field system will be built around three primary abstractions:

### 1.1 `Field` (The Mathematical Domain)
A `Field` is a resolution-independent function that maps spatial coordinates to vectors.
- **Interface:** `Field(positions: np.ndarray) -> np.ndarray` (where positions is `(N, 2)` and output is `(N, 2)`).
- **Composition:** Fields can be composed using standard arithmetic operators (`+`, `-`, `*`) and functional wrappers (e.g., `MaskedField`, `BlendedField`).
- **Determinism:** All noise-based fields must accept a `seed` and use a local `np.random.Generator`.

### 1.2 `StreamlineGenerator` (The Algorithm)
The generator responsible for integrating paths through a `Field`.
- **Input:** A `Field` and a set of `Strategies`.
- **Output:** A `StreamlinePattern`.
- **Integration:** Default to RK4 (Runge-Kutta 4th Order) for stability.

### 1.3 `StreamlinePattern` (The Semantic Data)
A collection of integrated curves with associated metadata (velocity, arc length, curvature).
- **Role:** Holds the semantic representation of the flow.
- **Rendering:** Converts to `Polyline` geometry for SVG output, supporting per-point or per-streamline styling.

## 2. Strategy-Driven Design

To improve on the existing interface, parameters will be passed via specialized strategy objects. This prevents "parameter bloat" in the main generator and makes behaviors explicitly swappable.

### 2.1 Strategy Types
- **`SeedingStrategy`:** Determines where streamlines begin (e.g., `RandomSeeding`, `GridSeeding`, `PoissonDiscSeeding`).
- **`TerminationStrategy`:** Determines when a streamline stops (e.g., `FixedLength`, `BoundaryTermination`, `ProximityTermination`).
- **`WidthStrategy`:** Controls stroke weight along the path (e.g., `UniformWidth`, `TaperedWidth`, `MagnitudeDrivenWidth`).
- **`ColorStrategy`:** Maps field or streamline properties to color (e.g., `AngleMappedColor`, `VelocityMappedColor`).

## 3. Integration with Existing Systems

### 3.1 Cross-Generator Influence
Fields are not restricted to streamlines. A `Field` can be passed as a `direction_guide` to `SpaceColonization` or other future generators, fulfilling the "Composability First" goal.

### 3.2 Rendering Consistency
`StreamlinePattern` will use the same `NetworkStyle` or a derived `StreamlineStyle` (Pydantic-based) to ensure that tapering, shading, and color mapping remain consistent with the branching patterns.

## 4. Technical Requirements

- **Spatial Indexing:** Use `scipy.spatial.cKDTree` for `ProximityTermination` and `PoissonDiscSeeding`.
- **Vectorization:** The `StreamlineGenerator` should integrate multiple seeds in parallel using NumPy broadcasting where possible.
- **Simplification:** Implement Ramer-Douglas-Peucker simplification to keep SVG file sizes manageable for long streamlines.

---

**Approval Required:** This direction formalizes the "Field as Function" and "Strategy-Driven Parameters" approach. Once approved, we will proceed to the detailed `change_requirements.md`.
