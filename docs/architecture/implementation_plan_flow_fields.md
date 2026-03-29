# Flow Fields Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use [executing-plans] mode to implement this plan task-by-task.

**Goal:** Implement a modular, strategy-driven Flow Field system for generating algorithmic streamlines and influencing other generators (like Space Colonization).

**Architecture:** A "Strategy-First" approach where a `FlowGenerator` composes swappable strategies for field definition, numerical integration, seeding, and termination. Fields are resolution-independent `(N, 2) -> (N, 2)` mappings.

**Tech Stack:** Python, NumPy, SciPy (KDTree), Pydantic, Shapely.

---

### Task 1: Core Field Abstraction

**Files:**
- Create: `src/bp_designs/core/field.py`
- Test: `tests/test_fields.py`

**Step 1: Define the Field ABC**
Implement the `Field` base class that supports composition via arithmetic operators.
```python
class Field(ABC):
    @abstractmethod
    def __call__(self, positions: np.ndarray) -> np.ndarray:
        """Map (N, 2) positions to (N, 2) vectors."""
        pass

    def __add__(self, other: Field) -> Field: ...
    # Add __sub__, __mul__ for composition
```

**Step 2: Implement Basic Fields**
- `ConstantField`: Returns a fixed vector.
- `RadialField`: Vectors point away from/towards a center.
- `NoiseField`: Uses `vnoise` or `opensimplex` with a local `np.random.Generator`.

**Step 3: Verification**
- Test: `tests/test_fields.py`
- Verify `NoiseField` is deterministic given a seed.
- Verify `RadialField` vectors at `(1, 0)` point away from `(0, 0)`.

---

### Task 2: Integration Strategies

**Files:**
- Create: `src/bp_designs/generators/flow/strategies.py`
- Test: `tests/test_flow_strategies.py`

**Step 1: Define IntegrationStrategy ABC**
```python
class IntegrationStrategy(ABC):
    @abstractmethod
    def step(self, field: Field, positions: np.ndarray, dt: float) -> np.ndarray:
        """Calculate next positions for N particles."""
        pass
```

**Step 2: Implement Euler and RK4**
- `EulerIntegrator`: `pos + field(pos) * dt`
- `RK4Integrator`: Standard 4th-order Runge-Kutta for stability.

**Step 3: Verification**
- Test: Integrate a `RadialField` (circular flow) for 100 steps.
- Verify `RK4Integrator` has lower drift (closer to starting radius) than `EulerIntegrator`.

---

### Task 3: Seeding & Termination Strategies

**Files:**
- Modify: `src/bp_designs/generators/flow/strategies.py`
- Test: `tests/test_flow_strategies.py`

**Step 1: Implement Seeding Strategies**
- `RandomSeeding`: Uniform distribution within a boundary.
- `GridSeeding`: Regular lattice of points.
- `PoissonDiscSeeding`: Uses `scipy.spatial.cKDTree` for blue-noise distribution.

**Step 2: Implement Termination Strategies**
- `FixedLength`: Stops after N steps or total distance.
- `BoundaryTermination`: Stops when leaving a `Polygon`.
- `ProximityTermination`: Stops when within `min_dist` of any existing streamline point (using `cKDTree`).

**Step 3: Verification**
- Verify `ProximityTermination` stops a streamline before it hits another.
- Verify `PoissonDiscSeeding` maintains minimum distance between seeds.

---

### Task 4: Flow Generator & Streamline Pattern

**Files:**
- Create: `src/bp_designs/generators/flow/generator.py`
- Create: `src/bp_designs/patterns/flow.py`
- Test: `tests/test_flow_generator.py`

**Step 1: Implement StreamlinePattern**
A semantic container for polylines and flow metadata (velocity, curvature).
```python
class StreamlinePattern(Pattern):
    streamlines: list[np.ndarray]  # List of (M, 2) arrays
    # Metadata for rendering
```

**Step 2: Implement FlowGenerator**
Orchestrates the loop: Seed -> Integrate -> Terminate -> Collect.
Ensure the integration loop is vectorized (integrating all active streamlines in one NumPy call).

**Step 3: Verification**
- Run `FlowGenerator` with `NoiseField` and `FixedLength`.
- Verify output is a `StreamlinePattern` with expected number of curves.

---

### Task 5: Rendering & Craft-Awareness

**Files:**
- Modify: `src/bp_designs/patterns/flow.py`
- Create: `src/bp_designs/patterns/network/renderer.py` (if needed for reuse)

**Step 1: Implement SVG Simplification**
Add Ramer-Douglas-Peucker (RDP) simplification to `StreamlinePattern.to_geometry()`.

**Step 2: Implement Styling Strategies**
- `TaperedWidthStrategy`: Matches `BranchNetwork` tapering.
- `AngleMappedColor`: Maps vector angle to a color palette.

**Step 3: Verification**
- Compare SVG file size with and without RDP simplification.
- Verify color mapping matches the field direction visually.

---

### Task 6: Integration with Space Colonization

**Files:**
- Modify: `src/bp_designs/generators/branching/strategies.py`
- Test: `tests/test_space_col_flow.py`

**Step 1: Implement FieldInfluenceGrowth**
A `GrowthStrategy` that blends the standard attraction vector with a `Field` vector.
```python
class FieldInfluenceGrowth(GrowthStrategy):
    def __init__(self, field: Field, weight: float = 0.5): ...
```

**Step 2: Verification**
- Run `SpaceColonization` with a `ConstantField` (e.g., "wind").
- Verify the tree leans in the direction of the field.

---

### Task 7: Final Verification Experiment

**Files:**
- Create: `src/experiments/flow_field_master.py`

**Step 1: Create a complex composition**
- Combine `NoiseField` and `RadialField`.
- Use `RK4Integrator` and `ProximityTermination`.
- Render with `TaperedWidthStrategy`.

**Step 2: Run and Review**
- Run: `poetry run python src/experiments/flow_field_master.py`
- Verify output in gallery.
