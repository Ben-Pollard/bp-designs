### Pattern Generator API
Every generator must implement:

```python
class PatternGenerator:
    def __init__(self, seed: int = 0, **params):
        """
        Args:
            seed: Random seed for determinism
            **params: Generator-specific parameters
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self) -> Geometry:
        """Generate pattern geometry.

        Returns:
            List of strokes (polylines as Nx2 numpy arrays)
        """
        pass
```

### Geometry Processors
Processors are pure functions:
```python
def process(geometry: Geometry, **params) -> Geometry:
    """Transform geometry without side effects."""
    # Return new geometry, don't modify input
    pass
```
