# Gallery & Experimentation Architecture

## Overview

The experimentation system enables systematic parameter exploration with interactive visual comparison. It consists of:

1. **Experiment Scripts** (Python) - Generate parameter grids and run experiments
2. **Experiment Outputs** (JSON + SVG) - Structured data from experiments
3. **Gallery Viewer** (HTML/JS) - Interactive front-end for viewing results

**Design Philosophy:** Separation of data generation (Python) and visualization (web). Gallery is data-driven and can display any experiment without modification.

---

## System Architecture

```
┌─────────────────────┐
│ Experiment Scripts  │ Python scripts in ./scripts/experiments/
│  (Python)           │ Define parameters, run generators
└──────────┬──────────┘
           │ generates
           ▼
┌─────────────────────┐
│ Experiment Outputs  │ Stored in ./output/experiments/[name]/
│  (JSON + SVG)       │ Structured, self-contained data
└──────────┬──────────┘
           │ reads
           ▼
┌─────────────────────┐
│ Gallery Viewer      │ Static HTML/JS in ./gallery/
│  (HTML/JS)          │ Data-driven, no hardcoded paths
└─────────────────────┘
```

---

## Directory Structure

```
bp-designs/
├── gallery/                    # Front-end viewer (static HTML/JS)
│   ├── index.html             # Main gallery page
│   ├── app.js                 # Vanilla JS application logic
│   └── styles.css             # Styling
│
├── scripts/                   # Experimentation code
│   └── experiments/           # Experiment runner scripts
│       ├── README.md          # How to use experiment framework
│       ├── space_colonization_basic.py
│       └── *.py               # More experiment scripts
│
├── output/                    # All generated outputs
│   └── experiments/           # Experiment results
│       ├── space_col_basic_001/
│       │   ├── config.json    # Experiment metadata
│       │   └── outputs/       # Generated variants
│       │       ├── var_0001.svg
│       │       ├── var_0001.json
│       │       └── ...
│       └── [experiment_name]/
│
└── src/bp_designs/
    └── experiment/            # Python API for experiments
        ├── runner.py          # ExperimentRunner class
        ├── space.py           # ParameterSpace class
        └── gallery.py         # (optional) Python gallery generator
```

---

## Data Schema

### Experiment Directory Structure

Each experiment creates a self-contained directory:

```
output/experiments/[experiment_name]/
├── config.json           # Experiment-level metadata
└── outputs/
    ├── var_0001.json    # Variant parameters
    ├── var_0001.svg     # Variant output
    ├── var_0002.json
    ├── var_0002.svg
    └── ...
```

### config.json Schema

```json
{
  "experiment_name": "space_col_basic_001",
  "timestamp": "2025-01-30T10:30:00Z",
  "description": "Basic parameter sweep of attraction_distance",
  "pattern_type": "SpaceColonization",
  "total_variants": 12,
  "successful": 12,
  "failed": 0,
  "parameters": {
    "varied": {
      "attraction_distance": [30, 50, 70],
      "segment_length": [1, 2, 3, 4]
    },
    "fixed": {
      "seed": 42,
      "width": 100,
      "height": 100
    }
  }
}
```

### Variant JSON Schema

```json
{
  "variant_id": "var_0001",
  "svg_path": "outputs/var_0001.svg",
  "params": {
    "attraction_distance": 30.0,
    "segment_length": 1.0,
    "num_attractions": 500,
    "seed": 42
  },
  "timestamp": "2025-01-30T10:30:15Z"
}
```

---

## Gallery Viewer Architecture

### Technology Stack

**Vanilla HTML/CSS/JS** - No framework, no build step

**Why vanilla?**
- Simple: Open `index.html` directly in browser
- AI-friendly: Standard web APIs, easy to understand and modify
- Future-ready: Can add frameworks later if needed
- No dependencies: Works offline, no npm/webpack

### Data Flow

```
1. User opens gallery/index.html
2. Gallery scans ./output/experiments/ for directories
3. Displays experiment list in dropdown
4. User selects experiment
5. Gallery loads config.json and variant JSONs
6. Renders variant grid with SVGs and metadata
```

### Key Features

- **Experiment Selection** - Dropdown to switch between experiments
- **Variant Grid** - Visual comparison of all variants
- **Metadata Display** - Parameters shown for each variant
- **Responsive Design** - Works on different screen sizes
- **No Hard-coded Paths** - Loads data dynamically

### Code Structure

```javascript
// app.js - ES6 module

// Data loading
async function loadExperiments() { ... }
async function loadExperiment(name) { ... }

// Rendering
function renderExperimentList(experiments) { ... }
function renderVariantGrid(variants, config) { ... }

// Event handlers
function onExperimentSelect(name) { ... }

// Init
document.addEventListener('DOMContentLoaded', init);
```

---

## Workflow

### 1. Running an Experiment

```bash
# Define parameters and run experiment
poetry run python scripts/experiments/my_experiment.py

# Output appears in:
# output/experiments/my_experiment_001/
```

### 2. Viewing Results

```bash
# Open gallery in browser
open gallery/index.html

# OR use Python server if needed (for file:// restrictions)
python -m http.server 8000
# Then open: http://localhost:8000/gallery/
```

### 3. Comparing Experiments

- Select different experiments from dropdown
- Compare parameter effects visually
- Document findings in `docs/exploration/`

---

## Future Evolution

### Phase 1: Viewer (Current)
- Display existing experiments
- Switch between experiments
- View variants and parameters

### Phase 2: Parameter Editor
- Edit parameters in UI
- Preview single variant
- Quick iteration without Python scripts

### Phase 3: Batch Generator
- Define parameter grids in UI
- Generate experiments from browser
- Full experimentation loop in front-end

### Technical Path Forward

**Phase 2 Requirements:**
- Add parameter input controls
- Call Python backend to generate single variant
- Consider: FastAPI backend or WASM Python?

**Phase 3 Requirements:**
- Grid definition UI
- Batch job management
- Progress tracking
- Consider: Web workers for parallelization?

---

## Python API (Programmatic Use)

The Python `bp_designs.experiment` module provides programmatic access:

```python
from bp_designs.experiment import (
    ParameterSpace,
    ExperimentRunner,
    generate_gallery,  # Optional: legacy HTML generator
)

# Define parameter space
space = ParameterSpace(...)
grid = space.to_grid()

# Run experiment
runner = ExperimentRunner(experiment_name="my_exp_001")
runner.run(grid=grid, generator_fn=my_generator)

# Gallery HTML is automatically at:
# output/experiments/my_exp_001/ (viewable in gallery/)
```

**Note:** The Python `generate_gallery()` function is deprecated in favor of the standalone HTML gallery viewer, but remains available for backward compatibility or custom HTML generation.

---

## Design Principles

### 1. Data-Driven
Gallery never hard-codes experiment names or paths. All data loaded dynamically.

### 2. Self-Contained Experiments
Each experiment directory contains everything needed to display it. No external dependencies.

### 3. Simple First
Use simplest tech that works. Add complexity only when needed.

### 4. Separation of Concerns
- Python: Data generation, computation
- Web: Visualization, interaction
- Clean interface: JSON files

### 5. Iteration Speed
Fast cycle: modify → run → view. No build steps, no compilation.

---

## Technical Considerations

### File:// Protocol Limitations

Modern browsers restrict `file://` access to local files for security. Solutions:

1. **Python HTTP Server** (Recommended for development)
   ```bash
   python -m http.server 8000
   ```

2. **Browser Flags** (Chrome)
   ```bash
   chrome --allow-file-access-from-files
   ```

3. **Production:** Deploy to static hosting (GitHub Pages, Netlify, etc.)

### CORS and Local Development

If adding backend API later:
- Use `localhost` with proper CORS headers
- Consider FastAPI with CORS middleware
- Or: Bundle everything as static files (no API calls)

### Performance Considerations

- **SVG Loading:** Lazy load variants as they scroll into view
- **Large Experiments:** Paginate or virtualize grid for 100+ variants
- **Image Caching:** Browser handles SVG caching automatically

---

## AI Development Guidelines

When modifying the gallery system:

1. **Read this doc first** - Understand architecture before making changes

2. **Maintain data schemas** - Don't break JSON contracts between Python and JS

3. **Keep gallery data-driven** - Never hard-code experiment names or paths

4. **Preserve simplicity** - Use vanilla JS unless complexity demands framework

5. **Test with real data** - Always test with actual experiment outputs

6. **Update this doc** - Keep architecture docs in sync with code

---

## Common Tasks

### Adding a New Experiment

1. Create script in `scripts/experiments/my_experiment.py`
2. Use `ExperimentRunner` API
3. Run script → output appears in `output/experiments/`
4. Open gallery → experiment appears in dropdown

### Modifying Gallery UI

1. Edit `gallery/index.html` (structure)
2. Edit `gallery/styles.css` (appearance)
3. Edit `gallery/app.js` (behavior)
4. Refresh browser

### Changing Data Schema

1. Update this doc with new schema
2. Update `ExperimentRunner` to write new format
3. Update `gallery/app.js` to read new format
4. Test with new and old experiments

---

## Questions & Decisions

### Why not React/Vue/Svelte?

- **Overkill** for current needs (displaying static data)
- **Complexity** adds build steps, dependencies, learning curve
- **AI-friendliness** vanilla JS easier to reason about
- **Can add later** if UI complexity grows

### Why not Flask/FastAPI backend?

- **Phase 1** doesn't need backend (just displaying files)
- **Static hosting** possible without server
- **Can add later** for Phase 2 (parameter editing, generation)

### Why separate gallery/ from src/?

- **Different concerns:** Python package vs. web viewer
- **Different deployment:** Package on PyPI, gallery on web host
- **Independence:** Gallery works without Python installed

### Why JSON schema instead of database?

- **Simplicity:** No DB setup, easy to inspect files
- **Portability:** Experiments are self-contained directories
- **Version control:** Can commit experiments to git if small
- **Future-proof:** Easy to migrate to DB if needed

---

## References

- `/scripts/experiments/README.md` - How to run experiments
- `/docs/exploration/LEARNINGS.md` - Parameter findings
- `/docs/api.md` - Pattern generator API
- `/docs/CLAUDE.md` - AI assistant instructions (includes gallery workflow)
