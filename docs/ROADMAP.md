# Development Roadmap

## Current Status: System Broken After Refactor

**Date:** 2025-01-30
**Situation:** Added new ABCs (Pattern, Generator) and rearranged folders. Tests failing, gallery not generating.
**Goal:** Get system working again - tests passing, gallery samples generating.

---

## Immediate Priority: Restore Working System

**See:** `docs/features/composition_spec.md` for architecture details.

### What's Been Done (Phase 2 Implementation)
- ✅ Core ABCs defined: `Pattern`, `Generator`, `CompositePattern`
- ✅ `BranchNetwork` refactored to implement `Pattern` interface
- ✅ `SpaceColonization` updated to accept guidance fields
- ✅ `PatternCombinator.guide()` and `PatternCombinator.texture()` implemented
- ✅ Field interface working (distance, depth, density, direction channels)

### What's Broken Now
After folder reorganization and ABC additions:
- [ ] Tests failing (need to update imports, paths)
- [ ] Gallery samples not generating
- [ ] May have broken existing `VoronoiTessellation` implementation

### Step-by-Step Recovery Plan

#### 1. Fix Core Imports & Structure (First Priority)
- [ ] Update all imports to reflect new folder structure
- [ ] Ensure `src/core/pattern.py` is importable
- [ ] Ensure `src/core/combinator.py` is importable
- [ ] Fix any circular import issues
- [ ] Verify `__init__.py` files are correct

#### 2. Get Basic Tests Passing
- [ ] Run test suite: `pytest tests/`
- [ ] Fix failing tests one module at a time:
  - [ ] Core pattern interface tests
  - [ ] BranchNetwork tests (field queries, geometry)
  - [ ] SpaceColonization tests (generation, determinism)
  - [ ] VoronoiTessellation tests (if they exist)
- [ ] Ensure determinism tests pass (same seed → same output)

#### 3. Validate Basic Pattern Generation
Create simple smoke test script:
```python
# scripts/smoke_test.py
from src.generators.branching.space_colonization import SpaceColonization

# Can we generate a basic tree?
gen = SpaceColonization(bounds=(0, 0, 100, 100), n_attractions=100, seed=42)
tree = gen.generate_pattern()
print(f"Generated tree with {len(tree.positions)} nodes")

# Can we query it as a field?
import numpy as np
points = np.array([[50, 50]])
distance = tree.sample_field(points, 'distance')
print(f"Distance at (50,50): {distance}")

# Can we render it?
geometry = tree.to_geometry()
print(f"Geometry has {len(geometry)} polylines")
```
- [ ] Run smoke test
- [ ] Fix any runtime errors

#### 4. Fix VoronoiTessellation Integration
- [ ] Check if `VoronoiTessellation` still works after refactor
- [ ] If broken, update to implement `Pattern` interface:
  - [ ] Add `sample_field()` method with channels
  - [ ] Add `available_channels()` method
  - [ ] Ensure `to_geometry()` still works
  - [ ] Add `bounds()` method
- [ ] Test Voronoi generation independently

#### 5. Validate Composition System
- [ ] Test `PatternCombinator.guide()`:
  ```python
  voronoi = VoronoiTessellation(...).generate_pattern()
  tree_gen = SpaceColonization(...)
  guided = PatternCombinator.guide(voronoi, tree_gen, 'boundary_distance')
  ```
- [ ] Test `PatternCombinator.texture()`:
  ```python
  tree = tree_gen.generate_pattern()
  textured = PatternCombinator.texture(tree, voronoi, threshold=5.0)
  geometry = textured.to_geometry()
  ```
- [ ] Verify results are deterministic

#### 6. Restore Gallery Generation
- [ ] Create simple gallery generation script:
  ```python
  # scripts/generate_basic_gallery.py
  # Generate 5-10 basic tree + voronoi combinations
  # Save as SVG files
  # Generate gallery HTML
  ```
- [ ] Run gallery generation
- [ ] Visually inspect outputs
- [ ] Ensure gallery displays correctly

#### 7. Document Current State
- [ ] Update `LEARNINGS.md` with any findings
- [ ] Note what works, what doesn't
- [ ] Document any architectural changes needed
- [ ] Mark this checkpoint in roadmap

---

## Success Criteria for Recovery

System is "working again" when:
- ✅ Test suite passes (`pytest tests/`)
- ✅ Smoke test runs without errors
- ✅ Can generate basic tree pattern
- ✅ Can generate basic Voronoi pattern
- ✅ Can combine patterns via `PatternCombinator.guide()`
- ✅ Gallery generates and displays 5+ example combinations
- ✅ All outputs are deterministic (same seed → same result)

---

## Next Steps After Recovery

Once system is working:

### Phase 2 Completion: Exploration & Documentation
- [ ] Generate systematic variations:
  - Guided growth (different influence strengths)
  - Textured patterns (different thresholds)
  - Parameter sweeps
- [ ] Use Gallery tool to batch render variations
- [ ] Identify 5-10 "natural-looking" compositions
- [ ] Document findings in `exploration/2025-01-30_tree_voronoi_composition.md`
- [ ] Update `LEARNINGS.md` with composition principles

### Missing Combinators (Optional)
- [ ] Implement `PatternCombinator.nest()` if needed for exploration
- [ ] Implement `PatternCombinator.blend()` if needed for exploration

---

## Decisions Log

### 2025-01-30: System refactor complete, now fixing breakage
- **What happened:** Implemented Phase 2 architecture (ABCs, field interface, combinators)
- **Current state:** Core implementation done but system broken after folder reorganization
- **Priority:** Get tests passing and gallery generating before continuing exploration

### 2025-01-30: Composition architecture implemented
- **Core components:**
  - `Pattern` ABC with field interface
  - `Generator` ABC with `generate_pattern()` method
  - `BranchNetwork` implements Pattern with 5 channels
  - `SpaceColonization` implements Generator with guidance support
  - `PatternCombinator` with semantic composition operators
  - `CompositePattern` for recursive composition
- **Reference:** See `docs/features/composition_spec.md` for full architecture

### 2025-01-30: Performance targets
- Field queries < 100ms for typical pattern sizes
- Vectorized operations using numpy
- Lazy KDTree initialization for spatial queries
- Deterministic generation (same seed → same output)

---

## Future Phases (Preserved from Original Roadmap)

### Phase 3: Pattern Refinement
- Add natural constraints (tapering, collision, boundaries)
- Find parameter "sweet spots" through systematic exploration
- Build preset library of proven configurations
- Validate against leather manufacturing constraints

### Phase 4: Additional Pattern Families
- Flow fields (directional texture)
- Reaction-diffusion (organic fill)
- Parametric curves (geometric motifs)
- Each implements Pattern interface → immediately composable

### Phase 5: Pattern Evaluation Framework
- Heuristics to estimate "naturalness"
- Manufacturing feasibility scoring
- Visual hierarchy analysis

### Phase 6: Composition Tooling
- Templates for layouts (border, corner, center, all-over)
- Spatial transforms (rotate, scale, repeat)
- Advanced masking and mutual influence
- Composition presets and recipes

### Phase 7: Library of "Moves"
- Document vocabulary of proven patterns
- Catalog composition recipes
- Build example gallery
- Create preset system

### Phase 8: Advanced Tooling
- Parameter editor in gallery
- Real-time preview
- Advanced knowledge management
- Export to manufacturing formats

---

## Key Principles

1. **Fix before extend:** Get system working before adding features
2. **Test continuously:** Every change should keep tests passing
3. **Visual validation:** Generate gallery samples frequently
4. **Determinism:** Same seed → same output (always)
5. **Manufacturing reality:** Every pattern must be manufacturable
6. **Document failures:** Dead ends teach us what doesn't work

---

## References

- **Architecture:** `docs/features/composition_spec.md` - Core interfaces, usage examples
- **Design Goals:** `docs/design_goals.md` - Philosophy and constraints
- **Learnings:** `docs/exploration/LEARNINGS.md` - What works, what doesn't
- **Resources:** `docs/resources/RESOURCES.md` - External references
