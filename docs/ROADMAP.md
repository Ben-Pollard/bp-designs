# Development Roadmap


### Phase 1: Generator detail and composition interface ###
- [ ] simple 2-d pattern generators: oval, rectangle, polygon.
- [ ] Integrate updated geometry api to get space colonisation working again
  - [ ] While implementing growth: accept initial and maximum bounds as polygons in place of width/height. Restrict attractor generation to intital bounds, expand intitial bounds around current nodes, kill out-of-bounds sources.
- [ ] Growth: vein size implementation, following original paper

###
- [ ] Redo Voronoi following code style and architecture practices learned from space col

### Phase 2 Completion: Exploration & Documentation
- [ ] Generate systematic variations:
  - Guided growth (different influence strengths)
  - Textured patterns (different thresholds)
  - Parameter sweeps
- [ ] Use Gallery tool to batch render variations
- [ ] Identify 5-10 "natural-looking" compositions
- [ ] Document findings in `exploration/2025-01-30_tree_voronoi_composition.md`
- [ ] Update `LEARNINGS.md` with composition principles

---


## Future Phases

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
