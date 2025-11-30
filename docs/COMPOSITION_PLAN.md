# Composition Exploration Plan

**Created:** 2025-01-30
**Status:** Planning Phase

---

## Objective

Understand how multiple patterns interact and compose to enable the overall design goals. We're not optimizing individual patterns yet—we're learning how composition works.

---

## Why Now?

**Current State:**
- Space colonization (branching) implemented and working
- Experimentation framework functional
- Ready to explore how patterns combine

**Design Goal:**
> "Patterns are not standalone images; they should be layerable, combinable, transformable, maskable, nestable. Every generator should be usable as a building block."

We can't achieve this goal with one pattern. We need at least two to understand composition mechanics.

---

## Second Pattern: Voronoi Tessellation

### Why Voronoi?

**1. Visual Contrast**
- Cellular vs. linear
- Fill vs. structure
- Organic cells vs. hierarchical branches
- Perfect complement to branching patterns

**2. Technical Simplicity**
- Well-understood algorithm (scipy.spatial.Voronoi)
- Deterministic and parameterizable
- Fast to implement
- Multiple visual variations (edges, cells, relaxed, weighted)

**3. Composition Potential**
- **As fill:** Voronoi cells inside branching boundaries
- **As mask:** Branches only appear within certain cells
- **As guide:** Branch growth influenced by cell centers
- **As texture:** Cellular background with branching foreground

**4. Manufacturing Compatibility**
- Natural cell sizes align with embossing constraints
- Clear positive/negative space
- Easily validated minimum spacing

**Alternatives Considered:**
- **Flow fields:** Great for composition, but more abstract; harder to validate visually
- **Reaction-diffusion:** Rich patterns, but complex implementation and slower iteration
- **L-systems:** Too similar to space colonization; won't teach us about heterogeneous composition

---

## Composition Modes to Explore

### 1. Layering (Simplest)
Generate two patterns independently, overlay in SVG.

**Questions:**
- How do line weights interact?
- What visual relationships emerge?
- Does one pattern dominate?
- Where should overlaps occur?

**Experiment:** Generate branching + Voronoi with various densities, view overlays.

### 2. Masking
One pattern defines regions where another appears.

**Questions:**
- Should branching grow only in certain cells?
- Should cells only appear where branches don't exist?
- How to handle boundaries?

**Experiment:** Use Voronoi cells as spatial bounds for branch growth.

### 3. Mutual Influence
Patterns affect each other's generation.

**Questions:**
- Can branch growth be biased toward cell centers?
- Can cell sites be placed along existing branches?
- What feedback loops are useful?

**Experiment:** Place Voronoi sites as branches grow, or grow branches toward cell centroids.

### 4. Sequential Application
Generate base pattern, then derive second pattern from it.

**Questions:**
- Can Voronoi be generated from branch endpoints?
- Can branches fill gaps in Voronoi structure?
- What emergent structures appear?

**Experiment:** Use branch nodes as Voronoi sites.

---

## API Considerations

### Current State
Each pattern generator:
- Takes parameters
- Returns `List[np.ndarray]` (polylines)
- Exports to SVG independently

### Composition Needs

**Option A: Simple Composition (Start Here)**
```python
# Generate separately, combine in export
branches = space_colonization.generate()
cells = voronoi.generate()

svg = SVGExporter()
svg.add_layer(cells, stroke="lightgray", stroke_width=0.5)
svg.add_layer(branches, stroke="black", stroke_width=1.0)
svg.save()
```

**Pros:** No API changes, easy to implement, explores visual relationships
**Cons:** No interaction between patterns

**Option B: Masked Generation**
```python
# One pattern defines spatial bounds for another
cells = voronoi.generate()
mask = create_mask_from_cells(cells, selected_indices=[0, 5, 10])

branches = space_colonization.generate(
    boundary_mask=mask  # Only grow in masked regions
)
```

**Pros:** True composition, patterns interact
**Cons:** Requires mask representation (polygon, raster, distance field?)

**Option C: Compositional Pipeline**
```python
# Patterns pass information via metadata
result1 = voronoi.generate()
result2 = space_colonization.generate(
    attraction_points=result1.metadata["cell_centers"]
)
```

**Pros:** Flexible, patterns can influence each other
**Cons:** Need richer return types (geometry + metadata)

### Recommended Approach

**Phase 1:** Start with Option A (simple layering)
- No API changes
- Fast iteration
- Learn visual principles

**Phase 2:** Add Option B (masking) when needed
- Identify which compositions require interaction
- Design minimal API extensions
- Implement mask representation

**Phase 3:** Generalize to Option C if patterns emerge
- Only if multiple patterns need similar metadata
- Build after understanding concrete use cases

---

## Implementation Plan

### Step 1: Implement Basic Voronoi Pattern
- [x] **(Implicit - not implemented yet)** Create `src/bp_designs/patterns/cellular/voronoi.py`
- [ ] Parameters:
  - `num_sites`: Number of cell centers
  - `relaxation_iterations`: Lloyd relaxation for organic cells
  - `seed`: Deterministic randomness
  - `render_mode`: "edges", "cells", "both"
  - `bounds`: (width, height)
- [ ] Output: Polylines representing cell edges
- [ ] Test determinism and basic constraints

### Step 2: Create Simple Composition Experiments
- [ ] Script: `scripts/experiments/composition_layering.py`
- [ ] Generate grids of:
  - Branching density × Voronoi density
  - Different stacking orders (branches over cells vs. cells over branches)
  - Different visual weights (thin/thick combinations)
- [ ] View in gallery, document observations

### Step 3: Explore Spatial Relationships
- [ ] Identify natural composition patterns from Step 2
- [ ] Experiment with:
  - Voronoi fill in branching gaps (use negative space)
  - Branching as structural frame, Voronoi as texture
  - Complementary densities (dense branches + sparse cells, etc.)

### Step 4: Document Composition Principles
- [ ] Update `docs/exploration/LEARNINGS.md` with findings:
  - What density ratios work?
  - Which stacking orders feel natural?
  - How do line weights affect hierarchy?
  - What spacing prevents visual conflict?
- [ ] Create composition presets (proven combinations)

### Step 5: Prototype Masking (If Needed)
- [ ] If simple layering isn't enough, implement basic masking:
  - Polygon-based bounds (select Voronoi cells, use as growth boundary)
  - Experiment with partial masks (grow in cells 0-N, not others)
  - Document API requirements

---

## Success Criteria

**We'll know this phase is successful when:**

1. **Visual Understanding**
   - We can articulate why certain compositions work
   - We have 5-10 proven composition examples
   - We understand hierarchy (which pattern should dominate when)

2. **Practical Knowledge**
   - Documented density ratios that prevent conflict
   - Line weight combinations that create clear visual hierarchy
   - Spacing rules for layered patterns

3. **API Insight**
   - Clear understanding of what API is needed for composition
   - No premature abstraction (only build what's proven necessary)
   - Simple, composable functions over complex frameworks

4. **Manufacturing Validation**
   - Composed patterns still meet physical constraints
   - Clear positive/negative space maintained
   - Minimum line thickness and spacing verified

---

## Open Questions

### Pattern Interaction
1. Should composition happen at geometry level or field level?
   - Geometry: Combine polylines (simpler, more direct)
   - Field: Combine scalar/vector fields before extraction (more flexible, more complex)

2. How to handle visual weight?
   - Different line thicknesses?
   - Different opacity/shading in SVG?
   - Different density?

3. What metadata is useful for composition?
   - Pattern bounds (bbox)?
   - Density information?
   - Hierarchical structure (which lines are primary/secondary)?
   - Negative space regions?

### API Design
4. Should patterns return rich objects or simple polylines?
   - Simple: `List[np.ndarray]` (current)
   - Rich: `PatternResult(geometry, metadata, bounds, ...)`
   - Decision: Start simple, enrich only when needed

5. How to represent masks?
   - Polygon lists?
   - Binary raster?
   - Signed distance fields?
   - Decision: Defer until we need masking

6. Should there be a `Composition` class?
   - Could handle layering, masking, transformations
   - Or just keep it functional? (`compose([pattern1, pattern2])`)
   - Decision: Functional first, class if patterns emerge

---

## Next Actions

**Immediate (This Session):**
1. Create this plan document ✓
2. Update ROADMAP.md with composition phase
3. Read or create Voronoi pattern stub

**Next Session:**
1. Implement basic Voronoi pattern generator
2. Create first composition experiment script
3. Generate initial layering variations
4. Review in gallery, document initial observations

---

## References

- Natural examples: Leaf veins (branching) + cellular structure
- Algorithmic: Nervous System studio's hybrid pattern systems
- Inspiration: Japanese patterns combining geometric and organic elements
- Resource: `docs/resources/algorithms.md` (Voronoi section - to be added)
