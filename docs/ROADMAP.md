# Development Roadmap


### Phase 1a: Generator detail and composition interface ###
- [x] simple 2-d pattern generators: oval, regular polygon.
- [x] Integrate updated geometry api to get space colonisation working again
- [x] While implementing growth: accept initial and maximum bounds as polygons in place of width/height. Restrict attractor generation to intital bounds
- [ ] expand intitial bounds around current nodes, kill out-of-bounds sources.
- [ ] Growth: vein size implementation, following original paper
- [x] pattern implements to_svg
- [x] line thickness / tapering
- [x] Organic rendering:
  - [x] Network refinement (decimation, relocation, subdivision)
  - [x] Outline-based skinning (polygons instead of lines)
- [x] organs / render styles
- [x] colour
- [ ] Multiple Starting points
- [x] Shading
  - [x] Pragmatic 2D SVG gradients implemented
  - [ ] Future: Move to 3D for advanced lighting
- [x] Generated organs
  - [x] Refactored to OrganPattern/OrganGenerator architecture
  - [x] Procedural blossoms (multi-ring, organic petal shapes)
- [x] Background
  - [x] Support for background_color in Canvas and to_svg
  - [x] Coupled background colors via ParameterSpace
  - [x] Refactor `BranchNetwork` into a modular package (`bp_designs.patterns.network`)

### Phase 1b: Composition refinement
- [x] Network shouldn't own the background
- [x] Generalisise layering so network can layer organs with background + border using same mechanism outside of network context
- [x] What is a canvas? Distinct from rectangle? What if we add background + border?
- [x] Abstract multi-item generation from experiment context to return items for layering
- [x] Top-level abstraction - canvases on canvases or just layered items?

### Phase 1c: Tree Fixes
- [x] Use thickness strategy
- [x] Use refinement
- [ ] Fix output exceeding background size 

### Phase 1d: Trees Experimentation
- [ ] Generate ideas
- [ ] Non-tree applications
  - [ ] Rivers (noise the growth vector)
  - [ ] Cracking
  - [ ] Leaf veins
- [ ] Experiment with size
- [x] Generate systematic variations:
- [x] Parallelize experiment runner for faster batch generation
- [ ] Guided growth (different influence strengths)
- [x] Use Gallery tool to batch render variations

### Phase 1c: Tree output
- [ ] For 2d colour printing: an array of trees



### Phase 2: Bring Voronoi up to standard
- [ ] Redo Voronoi following code style and architecture practices learned from space col

### Phase 3: Composition Experimentation & Further development
- [ ] Update `LEARNINGS.md` with composition principles
- [ ] Layering on canvas
- [ ] 3d


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
