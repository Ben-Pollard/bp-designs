# Plan: Fix Flow Joining Width Discontinuities

## Problem
Streamline joining results in width discontinuities when tapering is enabled. This is caused by:
1. The collision point being omitted from the merged streamline.
2. Incorrect ordering of points during merging (connecting target tip to source seed instead of source tip).
3. `TaperedWidth` strategy being sensitive to point ordering.

## Proposed Changes

### 1. `src/bp_designs/generators/flow/generator.py`
- Modify `generate_pattern` to pass the `next_position` and its magnitude to `_merge_streamlines`.
- Update `_merge_streamlines` signature to accept the collision point.
- Fix merge logic:
    - **Join to Start (target_index == 0):**
        - Result: `[source_seed, ..., source_tip, collision_point, target_start, ..., target_end]`
        - Logic: `source_pts + [collision_point] + target_pts`
    - **Join to End (target_index == target_len - 1):**
        - Result: `[target_start, ..., target_end, collision_point, source_tip, ..., source_seed]`
        - Logic: `target_pts + [collision_point] + source_pts[::-1]`

### 2. `src/bp_designs/generators/flow/strategies.py`
- Update `ProximityTermination.update_metadata` to handle the new merging logic correctly, ensuring spatial index IDs and indices remain valid for the merged streamline.

### 3. `tests/test_flow_joining.py`
- Add a test case that validates the width of segments across a join point when `TaperedWidth` is used.
- Verify that the number of points in the merged streamline is correct (source + target + 1).

## Verification Plan
1. Run existing tests: `poetry run pytest tests/test_flow_joining.py`
2. Run new tapering-specific test.
3. Generate a visual sample using `src/experiments/flow_fields/join_study.py` to confirm the fix visually.
