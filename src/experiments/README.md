# Experimentation Scripts

Systematic parameter exploration scripts for pattern generators.

## Quick Start

```bash
# Run a basic parameter sweep
poetry run python scripts/experiments/space_colonization_basic.py

# Run seed variations
poetry run python scripts/experiments/space_colonization_seeds.py

# View results
open experiments/space_col_basic_001/gallery.html
```

## How It Works

### 1. Define Parameter Space

```python
from bp_designs.experiment import ParameterSpace

space = ParameterSpace(
    name="my_exploration",
    ranges={
        # Explicit values
        "attraction_distance": [30.0, 50.0, 70.0],

        # Linear range: (min, max, num_steps)
        "segment_length": (1.0, 4.0, 4),  # [1.0, 2.0, 3.0, 4.0]

        # List of values
        "num_attractions": [100, 200, 500, 1000],
    },
    fixed={
        "seed": 42,  # Fixed across all variants
        "width": 100.0,
        "height": 100.0,
    },
)

grid = space.to_grid()  # Generate all combinations
```

### 2. Run Experiment

```python
from bp_designs.experiment import ExperimentRunner

def generate_pattern(params: dict):
    """Your generation function."""
    gen = SpaceColonization(**params)
    return gen.generate()

runner = ExperimentRunner(
    experiment_name="my_experiment_001",
    svg_width=100,
    svg_height=100,
    stroke_width=0.3,
)

runner.run(grid=grid, generator_fn=generate_pattern)
```

### 3. Generate Gallery

```python
from bp_designs.experiment import generate_gallery

generate_gallery(runner.exp_dir)
```

## Output Structure

```
experiments/
  my_experiment_001/
    config.json           # Experiment metadata, parameters, results
    outputs/
      var_0001.svg        # Generated pattern
      var_0001.json       # Parameters for this variant
      var_0002.svg
      var_0002.json
      ...
    gallery.html          # Interactive visualization
```

## Gallery Features

- **Grid layout** - Visual comparison of all variants
- **Metadata display** - Hover to see parameters
- **Failure tracking** - Failed variants logged in config.json
- **Responsive design** - Works on different screen sizes

## Creating New Experiments

1. Copy an existing experiment script
2. Modify parameter ranges for your exploration
3. Adjust fixed parameters as needed
4. Run and review gallery

## Tips

- **Start small** - Test with 4-9 variants before running full grids
- **Use fixed seed** - Keep seed constant when exploring other parameters
- **Track failures** - Check config.json for parameter combinations that failed
- **Name experiments** - Use descriptive names with version numbers (e.g., `space_col_density_003`)
- **Document findings** - Add observations to `docs/exploration/YYYYMMDD.md`

## Parameter Strategies

### Explore one parameter at a time
Fix all but one parameter to isolate effects:
```python
ranges={
    "attraction_distance": [20, 30, 40, 50, 60, 70, 80],
},
fixed={
    "num_attractions": 500,
    "segment_length": 2.0,
    "seed": 42,
}
```

### Explore interactions (2-3 parameters)
Test combinations of related parameters:
```python
ranges={
    "attraction_distance": [30, 50, 70],
    "kill_distance": [3, 5, 7],
    "segment_length": [1, 2, 4],
}
# 3 × 3 × 3 = 27 variants
```

### Seed diversity
See natural variation at fixed parameters:
```python
ranges={
    "seed": list(range(50)),  # 50 different seeds
},
fixed={
    # All other params fixed
}
```

## Next Steps

After exploring:
1. Document findings in `docs/exploration/LEARNINGS.md`
2. Create preset configs for "natural-looking" parameters
3. Add validated ranges as defaults to pattern classes
