# BP Designs - Algorithmic Pattern Generation

Generative system for creating natural, algorithmic patterns for physical fabrication.

**Core Philosophy:** Deterministic algorithms → organic emergence → physical craft

---

## Quick Start

### Prerequisites
- Python 3.12
- [Poetry](https://python-poetry.org/docs/#installation)

### Setup
```bash
# Clone and install
poetry install

# Run tests
poetry run pytest

# Format and lint
poetry run ruff format
poetry run ruff check
```

---

## Experiment Workflow

### 1. Run an Experiment
```bash
# Generate parameter variations
poetry run python src/experiments/regular_polygon_basic.py

# Output: output/experiments/[experiment_name]/
```

### 2. View Results
```bash
# Option A: Direct (if browser allows file:// access)
open gallery/index.html

# Option B: HTTP server
python -m http.server 8000
# Then: http://localhost:8000/gallery/
```

### 3. Iterate
- Select experiment from dropdown
- Compare parameter effects visually
- Document findings in `docs/exploration/[date]_[topic].md`
- Update `docs/exploration/LEARNINGS.md` with insights

**See:** `docs/ARCHITECTURE_GALLERY.md` for technical details

---

## Design Goals

### Algorithmic, Not AI
- Deterministic, rule-based generation (L-systems, space colonization, reaction-diffusion)
- No machine learning, no diffusion models
- Reproducible: same seed → same output

### Composability
Patterns are building blocks:
- Layerable
- Combinable
- Transformable
- Maskable

### Craft-Aware
Physical manufacturing constraints:
- Minimum line thickness
- Minimum spacing
- Adequate negative space
- Embossable depth and detail

**See:** `docs/design_goals.md` for full philosophy

---


## Working with Claude Code (AI Agent)

This project is designed for AI-assisted development. Claude reads project docs to understand context and help implement patterns.

### Session Startup
When starting a session, Claude will:
1. Read `docs/ROADMAP.md` for current priorities
2. Check `docs/exploration/*.md` for recent findings
3. Consult `docs/design_goals.md` for philosophy
4. Reference `docs/api.md` when writing pattern code

### Key Docs for Claude
- `CLAUDE.md` - Instructions for AI assistant (workflow, principles, examples)
- `ROADMAP.md` - Current phase, decisions, next steps
- `exploration/LEARNINGS.md` - Proven patterns and parameter ranges
- `ARCHITECTURE_GALLERY.md` - How experiments and gallery work

### Development Workflow
1. Explore in Jupyter notebooks (interactive prototyping)
2. Run experiments (systematic parameter sweeps)
3. View in gallery (compare variations)
4. Extract to modules (when stable)
5. Document findings (learnings, exploration logs)

**Philosophy:** Iteration and curation. Visual results matter more than code elegance (though both matter).

---

## Resources

### Documentation
- `docs/design_goals.md` - Aesthetic philosophy and constraints
- `docs/api.md` - Pattern generator API reference
- `docs/resources/algorithms.md` - Algorithm overviews
- `docs/resources/RESOURCES.md` - External papers and tools

### Examples
- `scripts/experiments/` - Example experiment scripts
- `output/experiments/` - Generated results (after running experiments)

---

## Testing

```bash
# Run tests
poetry run pytest

# With coverage
poetry run pytest --cov=src --cov-report=term-missing

# Run specific test
poetry run pytest tests/test_space_colonization.py
```

---
