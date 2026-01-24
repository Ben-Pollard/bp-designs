# Fix Plan: Semantic Mapping in BranchNetwork

The `IndexError` in `depths` reveals a fundamental issue: our `BranchNetwork` methods (like `depths`, `branch_ids`, and `decimate`) were incorrectly assuming that `parents` contains **indices** into the current arrays, when it actually contains **node IDs**.

## Root Cause
- `BranchNetwork.parents` stores `node_id` values.
- `depths` and `branch_ids` were using `self.parents[current]` as an index: `current = self.parents[current]`.
- This works only if `node_id == index`, which is true for the initial growth but breaks after decimation or subdivision.

## The Fix
We need to update all semantic traversal methods to use an `id_to_idx` mapping.

### 1. Update `depths`
```python
id_to_idx = {node_id: i for i, node_id in enumerate(self.node_ids)}
# ... inside loop ...
parent_id = self.parents[current_idx]
current_idx = id_to_idx[parent_id]
```

### 2. Update `branch_ids`
```python
id_to_idx = {node_id: i for i, node_id in enumerate(self.node_ids)}
# ... inside loop ...
parent_id = self.parents[current_idx]
current_idx = id_to_idx[parent_id]
```

### 3. Update `decimate`
The iterative decimation I implemented already uses `id_to_idx` for distance checks, but we must ensure the re-parenting logic correctly handles the ID-based nature of the `parents` array.

### 4. Update `subdivide`
Subdivision was also making index-based assumptions when creating `new_parents`. It needs to be verified for ID consistency.

## Verification
- Run `poetry run pytest tests/test_refinement.py`
- Run `poetry run pytest tests/test_space_colonization.py`
