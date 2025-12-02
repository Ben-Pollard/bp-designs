# Learnings - Distilled Knowledge

This file contains **distilled, high-level findings** from parameter exploration and experimentation.

**Purpose:** Quick reference for proven patterns, parameter ranges, and insights.

**Source:** Summarized from detailed `exploration/*.md` experiment logs.

**Update frequency:** After every 3-5 related experiments.

---

## How to Use This File

- **When starting pattern work:** Read relevant section to understand what's been proven
- **When exploring parameters:** Check for existing findings in similar space
- **When making decisions:** Consult "What Works" for validated approaches

---

## 1. Branching Patterns

### 1.1 Space Colonization

*Status: Not yet explored*

#### What Works
*(To be discovered)*

#### What Doesn't Work
*(To be discovered)*

#### Proven Presets
*(To be discovered)*

#### Key Insights
*(To be discovered)*

---

### 1.2 L-Systems

*Status: Not yet explored*

---

## 2. Flow Fields

*Status: Future exploration*

---

## 3. Reaction-Diffusion

*Status: Future exploration*

---

## 4. Tessellations (Voronoi, Delaunay)

*Status: Future exploration*

---

## 5. Composition Techniques

*Status: Future exploration*

### Layering
*(To be discovered)*

### Masking
*(To be discovered)*

### Layout Strategies
*(To be discovered)*

---

## 6. Processing Effects

*Status: Future exploration*

### Tapering
*(To be discovered)*

### Smoothing
*(To be discovered)*

### Collision Avoidance
*(To be discovered)*

---

## 7. General Principles

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

---

## 8. Architecture & Implementation

### Field-based Composition Architecture (2025-01-30)
- **Pattern Interface**: All patterns expose themselves as queryable spatial fields (`sample_field`, `available_channels`)
- **Generator Interface**: Generators produce patterns and can accept guidance fields for directed growth
- **CompositePattern**: Implements Composite design pattern, enabling recursive composition
- **PatternCombinator**: High-level semantic operations (guide, texture, blend, nest) that work on any Pattern pair via field queries
- **Key Insight**: Patterns maintain semantic structures internally but expose fields; combinators use field queries to implement high-level operations without knowing internal details
- **Determinism**: Same seed → same output (non-negotiable)
- **Import Structure**: Package layout: `src/bp_designs/core/` (interfaces), `src/bp_designs/generators/` (implementations), `src/bp_designs/patterns/` (data structures)
- **Experiment Module**: Separate `src/experiment/` module for parameter exploration and gallery generation
- **Gallery System**: Data-driven web gallery auto-discovers experiments via `gallery/experiments.json`

#### What Works
- Space colonization with guidance fields (Voronoi boundaries influence tree growth)
- Voronoi tessellation with multiple render modes (edges, cells, both)
- Basic composition operations: guide (successful), texture (basic rendering), blend (basic rendering)
- Experiment runner with parameter grids and automatic SVG generation

#### What Doesn't Work (Yet)
- Nest combinator not implemented
- Advanced rendering for texture and blend operations (currently returns component geometry)
- Performance optimization for large patterns

#### Key Insights
- Field-based abstraction enables universal composition without custom pairwise logic
- Keep experiment module separate from pattern generation for clean separation of concerns
- Use `poetry run python` to ensure correct Python version (>=3.12,<3.13)

---

## Maintenance Notes

**Last updated:** 2025-01-30 (architecture recovery and initial composition)

**Recent experiments distilled:**
- Architecture recovery: Field-based composition system implemented and tested
- Composition examples: 36 variants generated (guided, textured, blended)
- Gallery generation restored

**Next review:** After pattern parameter exploration
