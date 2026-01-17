# BP Designs - Architecture API

## Three-Layer System

```
Generator (algorithm) → Pattern (semantic) → Geometry (interchange)
```

---

## Pattern (Semantic Layer)

**Purpose:** Represents what the data means.

### Pattern ABC
```python
class Pattern(ABC):
    @abstractmethod
    def to_geometry(self) -> Geometry:
        """Convert to geometric representation."""
        pass

    @abstractmethod
    def bounds(self) -> tuple[float, ...]:
        """Bounding box."""
        pass
```

### Example Pattern Types
- **TreePattern** - Branching structures (nodes, parents, hierarchy)
- **CellularPattern** - Tessellations (cells, boundaries, adjacency)
- **ScalarField** - Continuous values over space (density, temperature)
- **VectorField** - Continuous directions over space (flow, gradients)

**Key:** Each pattern type has its own structure appropriate to its semantics.

---

## Geometry (Interchange Layer)

**Purpose:** Data containers with conversions to/from external libraries.

### Geometry ABC
```python
class Geometry(ABC):
    @abstractmethod
    def bounds(self) -> tuple[float, ...]:
        """Bounding box (minimal computation)."""
        pass

    # Add to_<library>() methods as needed, not preemptively
```

### Example Geometry Types
- **Polylines** - Line segments (tree branches, cell edges)
- **PointSet** - Points with properties (rasterized fields)
- **Polygon** - 2D regions (bounds, masks)
- **Mesh** - 3D surfaces (future)

### Design Rules
- ✅ Store data (numpy arrays, simple structures)
- ✅ Add `to_<library>()` conversions when needed
- ✅ Add `from_<library>()` constructors when needed
- ❌ Never implement geometric algorithms
- ❌ Never reimplement library functionality

**Why:** Switch libraries without breaking algorithms. Avoid maintenance burden.

---

## Generator (Algorithm Layer)

**Purpose:** Implements creation algorithms.

### Generator ABC
```python
class Generator(ABC):
    @abstractmethod
    def generate_pattern(self, **kwargs) -> Pattern:
        """Generate pattern. Subclasses define specific parameters."""
        pass
```

### Parameter Design
**Explicit named parameters show intent:**
- `bounds: Geometry` - Spatial region for generation
- `density_guide: ScalarField` - Modulate density
- `direction_guide: VectorField` - Bias direction
- Other inputs as needed (TreePattern, CellularPattern, etc.)

**Placement:**
- `__init__()` - Algorithm config (seed, distances, lengths)
- `generate_pattern()` - Input patterns/geometry, e.g. boundary, guidance field

### Example
```python
class SpaceColonization(Generator):
    def __init__(self, seed: int, kill_distance: float, segment_length: float):
        # Algorithm parameters
        pass

    def generate_pattern(
        self,
        bounds: Geometry,
        num_attractions: int,
        density_guide: ScalarField | None = None,
        direction_guide: VectorField | None = None
    ) -> TreePattern:
        # Use bounds for spatial containment
        # Use density_guide to modulate attraction placement
        # Use direction_guide to bias growth
        # Return TreePattern
        pass
```

---

## Data Flow Examples

### Coupled Parameters (Experimentation)
```python
space = ParameterSpace(
    name="organic_exploration",
    pattern={"organ_color": [Color.from_hex("#2d5a27")]},
    derived={
        "background_color": lambda p: p["organ_color"].complementary().to_hex()
    }
)
```

### Simple Generation
```python
bounds = Polygon.rectangle(0, 0, 100, 100)
gen = SpaceColonization(seed=42, kill_distance=5.0, segment_length=2.0)
tree = gen.generate_pattern(bounds=bounds, num_attractions=500)
geometry = tree.to_geometry()  # Polylines
svg = geometry.to_svg()
```

### Guided Generation
```python
density = ScalarField(...)  # Some density distribution
tree = gen.generate_pattern(
    bounds=bounds,
    num_attractions=500,
    density_guide=density
)
```

### Pattern as Input
```python
voronoi = VoronoiGenerator(...).generate_pattern(bounds=bounds)
voronoi_boundary = voronoi.to_geometry()  # Get geometric boundary
tree = gen.generate_pattern(
    bounds=voronoi_boundary,  # Use Voronoi cells as bounds
    num_attractions=300
)
```

### Field Conversion (when needed)
```python
def tree_to_density(tree: TreePattern) -> ScalarField:
    """Convert tree to density field for guiding another generator."""
    # Sample tree density on grid
    # Return ScalarField
    pass

tree1 = gen.generate_pattern(bounds=bounds, num_attractions=500)
density = tree_to_density(tree1)
tree2 = gen.generate_pattern(bounds=bounds, density_guide=density)
```

---

## Key Design Decisions

1. **Pattern types are specific** - TreePattern ≠ CellularPattern ≠ ScalarField
2. **Geometry is interchange** - Convert to libraries, don't reimplement
3. **Generator parameters are explicit** - Named roles, clear intent
4. **Bounds can be any Geometry** - Box, polygon, mesh, whatever
5. **Conversion is explicit** - Pattern → Field requires helper function
6. **Libraries are external** - Shapely, Open3D, Trimesh via conversions

---

## Common Patterns

### When to Create Pattern Type
- Represents distinct semantic structure
- Has domain-specific operations
- Examples: TreePattern (hierarchy), ScalarField (sampling)

### When to Create Geometry Type
- Need to represent new geometric primitive
- External library expects specific format
- Examples: PointSet (for Open3D), Mesh (for Trimesh)

### When to Add Conversion
- Actually need operations from that library
- Not before (YAGNI)
- Example: Add `to_shapely()` when you need containment tests

### When to Create Generator
- New algorithm for creating patterns
- Examples: SpaceColonization, VoronoiGenerator, FlowFieldGenerator

---

## Anti-Patterns (Don't Do)

❌ Generic `Pattern.sample_field()` on all types
❌ Geometry implementing algorithms (intersection, buffer, etc.)
❌ Generator with ambiguous `guidance_field` parameter
❌ One-size-fits-all Pattern class with polymorphic behavior
❌ Preemptively adding conversions "just in case"

---
