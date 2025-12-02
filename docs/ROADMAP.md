# Development Roadmap

## Current Status: System Recovered

**Date:** 2025-01-30
**Situation:** Phase 2 architecture (ABCs, field interface, combinators) implemented and tested. System now working.
**Goal:** Begin systematic exploration of composition parameters and pattern variations.

---

**Note:** Use `poetry run python` (or `poetry run pytest`) to ensure the correct Python version (>=3.12,<3.13) is used.

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
- [x] Tests failing (need to update imports, paths) - **FIXED**
- [x] Gallery samples not generating - **FIXED**
- [x] May have broken existing `VoronoiTessellation` implementation - **FIXED**

### Step-by-Step Recovery Plan

#### 1. Fix Core Imports & Structure (First Priority)
- [x] Update all imports to reflect new folder structure
- [x] Ensure `src/core/pattern.py` is importable
- [x] Ensure `src/core/combinator.py` is importable
- [x] Fix any circular import issues
- [x] Verify `__init__.py` files are correct

#### 2. Get Basic Tests Passing
- [x] Run test suite: `pytest tests/` (14/14 tests pass)
- [x] Fix failing tests one module at a time:
  - [x] Core pattern interface tests
  - [x] BranchNetwork tests (field queries, geometry)
  - [x] SpaceColonization tests (generation, determinism)
  - [x] VoronoiTessellation tests (if they exist)
- [x] Ensure determinism tests pass (same seed → same output)

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
- [x] Run smoke test (successful)
- [x] Fix any runtime errors

#### 4. Fix VoronoiTessellation Integration
- [x] Check if `VoronoiTessellation` still works after refactor (needed fixes)
- [x] If broken, update to implement `Pattern` interface:
  - [x] Add `sample_field()` method with channels (already present)
  - [x] Add `available_channels()` method (already present)
  - [x] Ensure `to_geometry()` still works (already present)
  - [x] Add `bounds()` method (already present)
- [x] Test Voronoi generation independently (successful)

#### 5. Validate Composition System
- [x] Test `PatternCombinator.guide()`:
  ```python
  voronoi = VoronoiTessellation(...).generate_pattern()
  tree_gen = SpaceColonization(...)
  guided = PatternCombinator.guide(voronoi, tree_gen, 'boundary_distance')
  ``` (successful)
- [x] Test `PatternCombinator.texture()`:
  ```python
  tree = tree_gen.generate_pattern()
  textured = PatternCombinator.texture(tree, voronoi, threshold=5.0)
  geometry = textured.to_geometry()
  ``` (successful, basic rendering implemented)
- [x] Verify results are deterministic (same seed → same output)

#### 6. Restore Gallery Generation
- [x] Create simple gallery generation script: `scripts/experiments/composition_examples.py` (36 variants)
- [x] Run gallery generation (36 successful, 0 failed)
- [x] Visually inspect outputs (SVGs generated in `output/experiments/composition_examples_001/outputs/`)
- [x] Ensure gallery displays correctly (gallery auto-discovers experiments via `gallery/experiments.json`)

#### 7. Document Current State
- [x] Update `LEARNINGS.md` with any findings (architecture section added)
- [x] Note what works, what doesn't (see LEARNINGS.md)
- [x] Document any architectural changes needed (noted)
- [x] Mark this checkpoint in roadmap (status updated above)

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

### 2025-01-30: Recovery complete - system working
- **Status:** All tests passing (14/14), gallery generation restored
- **Fixes applied:**
  - Updated import paths after folder reorganization
  - Fixed circular import issues using `TYPE_CHECKING`
  - Added backward compatibility for `VoronoiTessellation.generate()`
  - Fixed `SpaceColonization.generate_pattern()` to return `BranchNetwork` not geometry
  - Implemented basic rendering for `CompositePattern` texture and blend operations
  - Updated experiment scripts to use `sys.path.insert` for experiment module
  - Consolidated SVG export in `Geometry` class (removed separate export module)
- **Results:** Composition examples script generated 36 variants successfully
- **Next:** Begin systematic exploration of composition parameters

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
