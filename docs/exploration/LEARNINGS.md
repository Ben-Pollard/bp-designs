# Learnings - Distilled Knowledge


---

## General Principles

  Natural beauty comes from constrained emergence - algorithms operating
  under the right constraints, with the right parameters, composed with
  intention. Here's how we get there:

### Study Real Natural Objects First

  Before implementing anything, we need to understand what makes natural
  patterns natural:

  - Leaf veins: Hierarchical branching with tapering, angles between 30-45┬░,
   never crossing, denser near edges
  - Bark texture: Flow lines that avoid each other, local perturbations,
  consistent "grain"
  - Cellular structures: Roughly uniform distribution, no extreme size
  variation, relaxed (not jagged)


  ### Algorithms Need Constraints (Physics-Like Rules)

  A bare L-system can look mechanical. But add constraints:

  ##### Mechanical L-system
  branches grow at fixed angles
  all strokes same thickness
  perfect symmetry

  #### Natural L-system
  branches grow toward light sources (tropism)
  strokes taper based on hierarchy
  slight angle variation (┬▒5┬░)
  branches avoid crossing existing geometry
  growth stops near boundaries

  These constraints are the difference between algorithmic and natural.

  ### Parameter Spaces, Not Single Configs

  Each algorithm has a parameter space. Most of it produces garbage. Our job
   is to:

  1. Explore systematically - Generate grids of variations
  2. Identify "sweet spots" - Which parameter ranges feel natural?
  3. Document working ranges - Create presets/templates
  4. Build guardrails - Constrain parameters to known-good ranges


  ### Composition Has Rules Too

  Natural objects have structural hierarchy:

  - Base form: Overall shape (circle, elongated, organic blob)
  - Primary structure: Main veins, branches, flow lines
  - Secondary detail: Cellular infill, texture
  - Negative space: Areas of rest


  ### The "Almost" Principle

  Perfect = artificial. Natural = almost perfect:

  - Almost symmetric
  - Almost regular
  - Almost straight

*(Update as we discover general principles that apply across pattern families)*

--


#### Key Insights
- Field-based abstraction enables universal composition without custom pairwise logic
- Keep experiment module separate from pattern generation for clean separation of concerns
- Use `poetry run python` to ensure correct Python version (>=3.12,<3.13)
- **Network Refinement**: Raw algorithmic output (like Space Colonization) is often too jagged for physical fabrication. A three-step refinement process (Decimate → Subdivide → Relocate) significantly improves organic quality and manufacturability.
- **Organic Rendering (Outline-based Skinning)**: Moving from stroke-based rendering to a single unioned polygon "skin" (using Shapely) produces much cleaner vector paths for manufacturing. It handles complex branching points naturally and supports smooth tapering without overlapping artifacts.
- **Parameter Coupling**: Use a unified parameter space with namespacing (e.g., `network.num_attractions`, `render.bg_color`) instead of rigid categories. This allows any parameter to depend on any other via `derived` functions, ensuring perfect coupling across complex compositions.
- **Parameter Routing**: Experiment scripts should handle the routing of namespaced parameters to their respective generators using helpers like `split_params()`. This keeps the core generator logic clean and focused on the algorithm.
- **Distribution Strategies**: Organ distribution is a property of the network structure, not the organs themselves. Distribution strategies should be implemented where they can access the network's semantic information (like leaf nodes or branch hierarchy).

---
