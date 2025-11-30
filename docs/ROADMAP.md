# BP Designs - Development Roadmap

## Current Phase: Phase 1 - Deep Exploration of Branching Patterns

**Status:** Core Implementation Complete - Ready for Exploration
**Started:** 2025-01-23
**Goal:** Implement one pattern generator deeply, explore parameter space, learn what "natural" means in practice

---

## Why Start Here

Natural beauty comes from **constrained emergence**, not random algorithm chaining. We need to:
1. Deeply explore one algorithm's parameter space
2. Find "sweet spots" that feel natural
3. Build constraint systems (tapering, collision avoidance, boundary awareness)
4. Document what works and why
5. Establish patterns for future generators

This is craft + iteration + constraints, not just coding.

---

## Phase 1 Detailed Plan

### Step 1: Choose First Pattern Family
**Decision:** Start with **branching patterns**

### Step 2: Core Implementation
- [ ] Create test suite for determinism

### Step 3: Add Natural Constraints
- [ ] Stroke tapering (thick → thin based on hierarchy)
- [ ] Collision avoidance (branches don't cross)
- [ ] Boundary awareness (growth stops at edges)
- [ ] Optional: Tropism (growth bias toward direction)
- [ ] Optional: "Almost symmetry" perturbation

### Step 4: Parameter Exploration & Documentation
- [x] Generate 50-100 variations systematically
- [ ] Identify 5-10 "natural-looking" configurations
- [ ] Document parameter ranges that work
- [ ] Record insights in EXPLORATION_LOG.md
- [ ] Create preset library of known-good configs

### Step 5: Validate Against Physical Constraints
- [ ] Verify minimum line thickness
- [ ] Verify minimum spacing
- [ ] Check that patterns can emboss at shallow relief
- [ ] Ensure adequate negative space

---

## Decisions Log

### 2025-01-29: Branch continuity fix
- **Issue:** `BranchNetwork.to_geometry()` was creating discontinuous branches - shared trunk segments only belonged to first branch, creating visual gaps
- **Fix:** Changed geometry extraction to trace complete paths from each leaf to root, allowing nodes to appear in multiple branch polylines
- **Result:** Branches now render as continuous lines from root to leaves
- **Side effect:** Trunk segments are drawn multiple times (once per branch), creating natural line weight variation - keeping this as a feature option

### 2025-01-29: Experimentation framework complete
- **Features:** Failure tracking, variant metadata, responsive gallery display
- **Ready for:** Systematic parameter exploration and finding "natural" sweet spots

### 2025-01-30: Composition phase planning
- **Decision:** Move to Phase 2 (composition exploration) before optimizing individual patterns


---

## Open Questions

3. How to evaluate "natural" objectively? (User review vs metrics?)

---

## Phase 2: Composition Exploration (Next)

**Goal:** Understand how patterns interact and compose; learn what API is needed for composition.

**See:** `docs/COMPOSITION_PLAN.md` for detailed plan.

### Why Phase 2 Now?
- Core design goal is composability ("patterns are building blocks")
- Can't learn composition principles with one pattern
- No need to optimize individual patterns yet—need to understand interaction first

### Second Pattern: Voronoi Tessellation
**Choice:** Voronoi (cellular) over flow fields or reaction-diffusion
- Visual contrast with branching (cellular vs. linear)
- Simple to implement, fast iteration
- Clear composition potential (fill, mask, guide)
- Manufacturing-compatible

### Step 1: Implement Basic Voronoi
- [ ] Create `patterns/cellular/voronoi.py`
- [ ] Parameters: num_sites, relaxation_iterations, render_mode, bounds
- [ ] Output: Polylines (edges) compatible with existing export
- [ ] Test determinism

### Step 2: Explore Simple Layering
- [ ] Create `scripts/experiments/composition_layering.py`
- [ ] Generate parameter grids:
  - Branching density × Voronoi density
  - Different stacking orders
  - Different line weights
- [ ] View in gallery, identify what works

### Step 3: Document Composition Principles
- [ ] What density ratios prevent visual conflict?
- [ ] Which pattern should dominate when?
- [ ] How do line weights create hierarchy?
- [ ] Update `LEARNINGS.md` with findings

### Step 4: Explore Masking (If Needed)
- [ ] If layering isn't sufficient, implement spatial masking
- [ ] Experiment: branches only grow in selected Voronoi cells
- [ ] Document API requirements for masking

### Step 5: Design Composition API
- [ ] Based on experiments, determine what API is needed
- [ ] Start simple (functional composition over complex classes)
- [ ] Only build what's proven necessary
- [ ] Create composition presets (proven combinations)

### Success Criteria
- 5-10 proven composition examples
- Documented principles (density ratios, hierarchy, spacing)
- Clear understanding of required API (no premature abstraction)
- Compositions meet manufacturing constraints

---

## Future Phases (Outline)

### Pattern Evaluation
Basic pattern evaluation framework. Start with simple heuristics to estimate beauty. It's a form of validation that should extend the manufacturing constraints framework.

### Phase Pattern Refinement
After understanding composition, return to optimize individual patterns:
- Add natural constraints (tapering, collision avoidance, boundary awareness)
- Find parameter "sweet spots" through systematic exploration
- Build preset library of proven configurations
- Validate against leather constraints

### Phase Composition Tooling
- Implement composition API based on Phase 2 learnings
- Templates for common layouts (border, corner, center)
- Advanced masking and mutual influence
- High-level composition functions

### Phase Additional Pattern Families
- Flow fields (directional texture)
- Reaction-diffusion (organic fill)
- Parametric curves (geometric motifs)
- Apply composition principles from Phase 2

### Phase Library of "Moves"
- Document vocabulary of proven patterns
- Catalog composition recipes
- Build curated example gallery
- Create pattern/composition presets

### Advanced Tooling
- Interactive parameter editor in gallery
- Real-time preview
- Advanced knowledge management for research (graph RAG, embeddings)

---
