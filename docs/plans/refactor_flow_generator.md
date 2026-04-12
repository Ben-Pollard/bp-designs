# Plan: Refactor FlowGenerator for Maintainability

## Goal
Clean up the `FlowGenerator` implementation to eliminate nested loops and complex conditionals, making it easier to debug and extend. Align the architecture with the `SpaceColonization` generator.

## Proposed Changes

### 1. Internal State Management
- Define a `_Streamline` helper class to hold:
    - `id`: Unique identifier
    - `points`: List of `np.ndarray`
    - `magnitudes`: List of `float`
    - `is_active`: Boolean flag

### 2. `FlowGenerator` Refactoring
- **`generate_pattern`**: High-level loop that calls `_execute_step`.
- **`_initialize_generation`**: 
    - Generate seeds.
    - Create `_Streamline` objects.
    - Initialize `TerminationStrategy`.
- **`_execute_step`**:
    - Get current positions of active streamlines.
    - Call `_get_steered_positions`.
    - Call `_handle_collisions`.
    - Update active streamlines with new points.
- **`_get_steered_positions`**:
    - Pure logic for blending field vectors with nearby tangents and pull vectors.
- **`_handle_collisions`**:
    - Check `should_terminate`.
    - If join strategy exists, check `should_join`.
    - Return a mapping of streamline ID to action (Continue, Terminate, Join).
- **`_merge_streamlines`**:
    - Handle the "tip-to-tip" connection logic.
    - Update `ProximityTermination` metadata.

### 3. Consistency
- Ensure `StreamlinePattern` remains the output type.
- Use similar naming conventions and method structures as `SpaceColonization`.

## Verification Plan
1. Run `poetry run pytest tests/test_flow_joining.py` to ensure no regressions in joining logic.
2. Run `poetry run python src/experiments/flow_fields/join_study.py` to verify visual parity.
