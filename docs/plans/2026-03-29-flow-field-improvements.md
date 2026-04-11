# Flow Field Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use [executing-plans] mode to implement this plan task-by-task.

**Goal:** Enhance the flow field system with dynamic termination, modular rendering strategies, performance optimizations, and magnitude-driven styling.

**Architecture:** 
- **Vectorization:** Shift from point-wise to batch processing in `NoiseField` and `FlowGenerator`.
- **Strategy Pattern:** Decouple rendering logic (width/color) from the `StreamlinePattern` class into swappable strategy objects.
- **Dynamic State:** Allow `ProximityTermination` to update its spatial index during generation.

**Tech Stack:** Python, NumPy, SciPy (cKDTree), Shapely, vnoise.

---

### Task 1: Performance - Vectorize NoiseField

**Files:**
- Modify: `src/bp_designs/core/field.py`

**Step 1: Optimize NoiseField.__call__**
Replace the list comprehension with a more efficient approach.

**Step 2: Verify with benchmark**
Run a small script to compare performance with 10,000 points.

---

### Task 2: Performance - Vectorize FlowGenerator Loop

**Files:**
- Modify: `src/bp_designs/generators/flow/strategies.py`
- Modify: `src/bp_designs/generators/flow/generator.py`

**Step 1: Update TerminationStrategy ABC**
Change `should_terminate` to return `np.ndarray[bool]`.

**Step 2: Update implementations**
Update `FixedLengthTermination`, `BoundaryTermination` (using `shapely.vectorized`), and `ProximityTermination`.

**Step 3: Refactor FlowGenerator loop**
Remove the per-streamline termination check loop in favor of the boolean mask.

---

### Task 3: Dynamic Proximity Termination

**Files:**
- Modify: `src/bp_designs/generators/flow/strategies.py`
- Modify: `src/bp_designs/generators/flow/generator.py`

**Step 1: Add update() to ProximityTermination**
Allow adding points to the internal `cKDTree`.

**Step 2: Update FlowGenerator**
Call `termination_strategy.update(next_pos)` at each step if it's a proximity strategy.

---

### Task 4: Rendering Strategy Refactor

**Files:**
- Modify: `src/bp_designs/patterns/flow.py`

**Step 1: Define WidthStrategy and ColorStrategy ABCs**
Create the base classes and standard implementations (`ConstantWidth`, `TaperedWidth`, `ConstantColor`, `AngleColor`).

**Step 2: Update StreamlinePattern.render**
Refactor to accept these strategies instead of hardcoded logic.

---

### Task 5: Magnitude Mapping

**Files:**
- Modify: `src/bp_designs/generators/flow/generator.py`
- Modify: `src/bp_designs/patterns/flow.py`

**Step 1: Store magnitudes in FlowGenerator**
Compute `np.linalg.norm(field_vals)` during integration and store in `StreamlinePattern`.

**Step 2: Implement MagnitudeWidth and MagnitudeColor**
Create strategies that use the stored magnitudes to drive visual properties.

---

### Task 6: Verification

**Files:**
- Create: `tests/test_flow_improvements.py`

**Step 1: Write tests for dynamic termination**
**Step 2: Write tests for magnitude mapping**
**Step 3: Run all tests**
