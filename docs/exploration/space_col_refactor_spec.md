# Technical Specification: Space Colonization Strategy Refactor

## 1. Objective
Refactor the `SpaceColonization` generator to support modular "behaviors" via strategy injection. This allows for artistic variations (momentum, grid-snapping, moving attractors, graph topology) without bloating the core algorithm.

## 2. Proposed Architecture

### A. Core Interfaces (`src/bp_designs/generators/branching/strategies.py`)

These strategies are designed to be **highly general**, allowing us to implement almost all the "wild" ideas by combining different strategy implementations.

#### 1. `GrowthStrategy` (The "How")
Controls the movement and physics of the branches.
*   **Generalizes:** *Living Calligraphy* (momentum), *Crystalline* (grid-snapping), *Gravitational Lensing* (warping vectors), *Tropism* (global bias).
*   **Interface:**
    ```python
    class GrowthStrategy(ABC):
        @abstractmethod
        def refine_vectors(self, vectors: np.ndarray, network: BranchNetwork) -> np.ndarray:
            """Modify raw attraction vectors (e.g., add inertia, snap to grid)."""
            pass
    ```

#### 2. `AttractionStrategy` (The "Where")
Controls the lifecycle and behavior of the targets.
*   **Generalizes:** *Boids* (moving targets), *Resource Scarcity* (depleting points), *Parasitic* (clinging to surfaces), *Noise-Driven Density*.
*   **Interface:**
    ```python
    class AttractionStrategy(ABC):
        @abstractmethod
        def initialize(self, num: int, boundary: Polygon) -> np.ndarray:
            """Initial placement (e.g., uniform, noise-based, or on-surface)."""
            pass
        
        @abstractmethod
        def update(self, attractions: np.ndarray, network: BranchNetwork) -> np.ndarray:
            """Update points per iteration (e.g., movement, consumption)."""
            pass
    ```

#### 3. `TopologyStrategy` (The "Structure")
Controls the connectivity and relationship between nodes.
*   **Generalizes:** *Neural Mats* (fusion/cross-linking), *Erosion* (pruning), *Recursive* (seeding new networks).
*   **Interface:**
    ```python
    class TopologyStrategy(ABC):
        @abstractmethod
        def extend(self, network: BranchNetwork, new_positions: np.ndarray, parents: np.ndarray) -> BranchNetwork:
            """Define how new nodes connect (e.g., tree vs. graph)."""
            pass
    ```

### B. Default Implementations
*   `DefaultGrowth`: Standard normalized vector * segment length.
*   `DefaultAttraction`: Static points, rejection sampling in boundary.
*   `DefaultTopology`: Standard parent-child tree structure.

### C. "Wild" Implementations (Initial Set)
*   `MomentumGrowth`: Blends current growth vector with the previous segment's direction.
*   `GridSnappedGrowth`: Snaps vectors to 45/90 degree increments.
*   `BoidsAttraction`: Attraction points move using flocking logic each iteration.
*   `FusionTopology`: Checks for nearby existing nodes and "fuses" (cross-links) instead of always creating new nodes.

## 3. Implementation Plan

1.  **Define Strategy ABCs:** Create `strategies.py` with the base classes.
2.  **Refactor `SpaceColonization.__init__`:** Accept strategy instances (defaulting to standard behavior).
3.  **Refactor `SpaceColonization._iterate`:**
    *   Call `attraction_strategy.update()` at start of loop.
    *   Call `growth_strategy.refine_vectors()` after computing raw directions.
    *   Call `topology_strategy.extend()` to produce the updated network.
4.  **Update `BranchNetwork`:** Ensure it can handle non-tree topologies (graph edges) if `FusionTopology` is used.
5.  **Migration:** Ensure existing experiments (`grid.py`, `big_trees.py`) still work with the new defaults.

## 4. Verification
*   Run `tests/test_space_colonization.py` to ensure no regressions in standard growth.
*   Create a new experiment `src/experiments/wild_strategies_test.py` to demonstrate a "Momentum + Grid" hybrid.
