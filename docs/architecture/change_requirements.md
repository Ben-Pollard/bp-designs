# Flow Fields — Change Requirements (Revised)

This document translates the agreed architectural direction into actionable technical requirements, emphasizing a **Strategy-First** approach where all behavioral and structural parameters are passed via swappable strategies.

## 1. Core Architecture: The Strategy-First Flow Generator

Instead of a monolithic generator with many parameters, we will implement a `FlowGenerator` that acts as a compositor for specialized strategies.

### 1.1 `bp_designs.generators.flow.FlowGenerator`
The primary entry point for flow-based patterns.
- **Role:** Orchestrates the integration loop.
- **Strategies:**
    - `field_strategy`: Defines the vector field (Noise, Trig, Radial).
    - `integration_strategy`: Defines the numerical method (Euler, RK4).
    - `seeding_strategy`: Defines where streamlines/particles originate.
    - `termination_strategy`: Defines when a streamline stops.
    - `refinement_strategy`: (Optional) Post-processes the resulting curves (Simplification).

### 1.2 Strategy ABCs (`bp_designs.generators.flow.strategies`)

| Strategy Type | Responsibility | Example Implementations |
|---------------|----------------|-------------------------|
| **`FieldStrategy`** | `(N, 2) -> (N, 2)` vector mapping | `NoiseField`, `TrigonometricField`, `RadialField`, `CompositeField` |
| **`IntegrationStrategy`** | `(pos, field, step) -> next_pos` | `EulerIntegrator`, `RK4Integrator` |
| **`SeedingStrategy`** | `(boundary, count) -> (M, 2)` seeds | `RandomSeeding`, `GridSeeding`, `PoissonDiscSeeding` |
| **`TerminationStrategy`** | `(state) -> bool` stop condition | `FixedLength`, `BoundaryTermination`, `ProximityTermination` |

## 2. Implementation Steps

### Step 1: Field & Integration Foundations
1. Implement `FieldStrategy` ABC and `NoiseField` (using `vnoise` or similar).
2. Implement `IntegrationStrategy` ABC with `RK4Integrator` as the high-fidelity default.
3. **Benefit:** This allows any field to be used with any integrator immediately.

### Step 2: The Flow Generator & Streamline Pattern
1. Implement `FlowGenerator` which takes the above strategies in its `__init__`.
2. Implement `StreamlinePattern` in `src/bp_designs/patterns/flow/`.
3. **Benefit:** The generator remains "pure" and doesn't need to know *how* the field is calculated or *how* integration works.

### Step 3: Seeding & Termination (The "Streamline" Logic)
1. Implement `SeedingStrategy` (starting with `RandomSeeding`).
2. Implement `TerminationStrategy` (starting with `FixedLength`).
3. Implement `ProximityTermination` using `scipy.spatial.cKDTree` to prevent streamlines from crowding (as requested in design doc).

### Step 4: Rendering & Style
1. Implement `StreamlineStyle` (Pydantic model) for `render_params`.
2. Support `TaperedWidthStrategy` and `AngleMappedColor` within the rendering logic.

## 3. Verification & Alignment

### 3.1 Consistency with `SpaceColonization`
This approach mirrors the `GrowthStrategy` / `AttractionStrategy` pattern but applies it more "deliberately" by moving *all* core logic into strategies.

### 3.2 Composability
Because `FieldStrategy` is a standalone interface, a `NoiseField` can be passed to `SpaceColonization` as a `direction_guide` without modification.

### 3.3 Verification Criteria
- **Strategy Swapping:** Verify that switching from `EulerIntegrator` to `RK4Integrator` produces a smoother, more accurate circle in a `RadialField`.
- **Proximity Termination:** Verify that streamlines stop before intersecting when `ProximityTermination` is active.
- **Determinism:** Ensure the `seed` is passed through the `FlowGenerator` to all strategies (Field, Seeding).

---

**Note on "Algorithms as Strategies":** In this design, the "Algorithm" (Streamline generation) is the composition of these strategies. This allows us to later implement a `ParticleSnapshotGenerator` that reuses the `FieldStrategy`, `IntegrationStrategy`, and `SeedingStrategy`, but ignores `TerminationStrategy`.
