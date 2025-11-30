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
**Decision:** Start with **branching patterns** (space colonization or L-systems)
- Most fundamental to natural forms (veins, roots, trees)
- Well-understood algorithms
- Clear visual feedback
- Good test case for constraint systems

**Next decision needed:** Space colonization vs L-systems?
- Space colonization: More organic, less mechanical
- L-systems: More control, easier parameterization

### Step 2: Core Implementation
- [x] Implement base algorithm with extensive parameters
- [x] Add basic SVG export for visual review
- [x] Create Jupyter notebook for interactive exploration
- [x] Build parameter explorer (generate grids of variations)
- [ ] Create test suite for determinism

### Step 3: Add Natural Constraints
- [ ] Stroke tapering (thick → thin based on hierarchy)
- [ ] Collision avoidance (branches don't cross)
- [ ] Boundary awareness (growth stops at edges)
- [ ] Optional: Tropism (growth bias toward direction)
- [ ] Optional: "Almost symmetry" perturbation

### Step 4: Parameter Exploration & Documentation
- [ ] Generate 50-100 variations systematically
- [ ] Identify 5-10 "natural-looking" configurations
- [ ] Document parameter ranges that work
- [ ] Record insights in EXPLORATION_LOG.md
- [ ] Create preset library of known-good configs

### Step 5: Validate Against Leather Constraints
- [ ] Verify minimum line thickness (0.3-0.5mm)
- [ ] Verify minimum spacing (0.6-0.8mm)
- [ ] Check that patterns can emboss at shallow relief
- [ ] Ensure adequate negative space (30-40%)

---

## Decisions Log

### 2025-01-23: Project structure and exploration strategy
- Created ROADMAP.md for active development tracking
- Decided on iterative exploration approach over "implement everything"
- Established that exploration and implementation are separate concerns
- Updated CLAUDE.md with session startup routine

### 2025-01-23: Architecture decisions
- **Geometry:** Start simple with `List[np.ndarray]`, add richness later only if needed
- **Exploration:** Jupyter-first workflow for interactive experimentation
- **Visualization:** SVG export with inline display + batch HTML gallery generation from notebooks
- **Testing:** Unit tests for determinism and constraints (no snapshot testing)
- **First pattern:** Space colonization (more organic than L-systems)

### 2025-01-23: Knowledge management restructure (Option C)
- **Trimmed CLAUDE.md:** Reduced from ~2000 to ~500 tokens by removing overlap with DESIGN_NOTES.md
- **Updated session startup:** Read ROADMAP.md always, other files selectively based on current work
- **Created exploration/:** Directory for detailed experiment logs (not read automatically)
- **Created LEARNINGS.md:** Distilled findings from experiments (high-level reference)
- **Token savings:** ~70% reduction in routine context (from ~5300 to ~1600 tokens per session)
- **Principle:** Dashboard → Details (read only what's needed)

### 2025-01-29: Branch continuity fix
- **Issue:** `BranchNetwork.to_geometry()` was creating discontinuous branches - shared trunk segments only belonged to first branch, creating visual gaps
- **Fix:** Changed geometry extraction to trace complete paths from each leaf to root, allowing nodes to appear in multiple branch polylines
- **Result:** Branches now render as continuous lines from root to leaves
- **Side effect:** Trunk segments are drawn multiple times (once per branch), creating natural line weight variation - keeping this as a feature option

### 2025-01-29: Experimentation framework complete
- **Components:**
  - `ParameterSpace` - Define parameter ranges (explicit lists or linear ranges)
  - `ParameterGrid` - Generate all combinations via cartesian product
  - `ExperimentRunner` - Execute experiments, save SVGs + metadata
  - `generate_gallery()` - Create interactive HTML gallery with parameter display
- **Workflow:** Script-based (not Jupyter) - define params → run → view gallery
- **Output:** `experiments/{name}/outputs/` (SVGs + JSON), `gallery.html`, `config.json`
- **Features:** Failure tracking, variant metadata, responsive gallery display
- **Ready for:** Systematic parameter exploration and finding "natural" sweet spots

---

## Open Questions

3. How to evaluate "natural" objectively? (User review vs metrics?)
4. ~~Should we build visualization tools first or iterate in code?~~ **Decided:** Build tools, iterate in Jupyter

---

## Future Phases (Outline)

### Experimentation Tooling
- [x] Gallery viewer (HTML with grid layout)
- [x] Save outputs with metadata (JSON + SVG)
- [x] Organization of experiment code (scripts/experiments/)
- [x] Parameter space definition for multi-generation (ParameterSpace + ParameterGrid)
- [ ] Interactive experiment GUI (future enhancement)
- [ ] Gallery filtering/sorting (future enhancement)
- [ ] Export metadata for compositional analysis

### Second Pattern Family
- Flow fields or reaction-diffusion
- Apply learnings from Phase 1
- Build composition examples

### API Composition
- Work out how to combine patterns
- Simple chaining, like image processor workflow, or metadata/composition aware?
- What changes to the API would be needed?

### Image Composition Patterns
- Study how branching + texture combine in nature
- Implement composition templates (border, corner, center)
- Test layering and masking
- Create high-level composition API

### Library of "Moves"
- Document vocabulary of proven patterns
- Build example gallery

### Advanced Knowledge Management
Exploration of:
- Local Graph RAG system
- Small local LLM for embeddings
- MS GraphRAG library
- Knowledge graph front-end

---
