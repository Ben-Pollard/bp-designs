# Network Refinement: Decimation, Relocation, and Subdivision

To achieve organic, smooth branching patterns, the raw output of the Space Colonization algorithm requires refinement. This document outlines the algorithms for node decimation, relocation (smoothing), and subdivision.

## 1. Node Decimation
Decimation reduces the complexity of the network by removing redundant nodes.

### Algorithm: Distance-based Decimation
1.  Iterate through all nodes $i$ (except roots).
2.  Calculate distance $d = \|P_i - P_{parent(i)}\|$.
3.  If $d < \epsilon$:
    -   For all children $j$ of $i$, set $parent(j) = parent(i)$.
    -   Remove node $i$.

### Algorithm: Angle-based Decimation
1.  Iterate through nodes $i$ with exactly one child $j$.
2.  Calculate the angle $\theta$ between vectors $\vec{v_1} = P_i - P_{parent(i)}$ and $\vec{v_2} = P_j - P_i$.
3.  If $\theta < \text{threshold}$ (i.e., segments are nearly collinear):
    -   Set $parent(j) = parent(i)$.
    -   Remove node $i$.

## 2. Node Relocation (Smoothing)
Smoothing reduces the jaggedness of the growth process.

### Algorithm: Laplacian Smoothing
For each node $i$:
1.  Identify neighbors $N(i) = \{parent(i)\} \cup \{children(i)\}$.
2.  If $i$ is a root or a leaf, optionally keep it fixed to preserve the overall structure.
3.  Calculate new position:
    $P_i' = (1 - \alpha) P_i + \alpha \frac{\sum_{j \in N(i)} P_j}{|N(i)|}$
    where $\alpha \in [0, 1]$ is the smoothing factor.

## 3. Subdivision
Subdivision increases the resolution of the network, allowing for smoother curves after subsequent relocation.

### Algorithm: Midpoint Subdivision
For each segment $(P, C)$ where $P = parent(C)$:
1.  Create a new node $M$.
2.  Set $P_M = \frac{P_P + P_C}{2}$.
3.  Update relationships:
    -   $parent(M) = P$
    -   $parent(C) = M$
4.  Interpolate other properties (timestamp, thickness) if necessary.

## Proposed API for `BranchNetwork`

```python
class BranchNetwork:
    def decimate(self, min_distance: float = 1.0) -> BranchNetwork:
        """Remove nodes that are too close to their parents."""
        ...

    def relocate(self, alpha: float = 0.5, iterations: int = 1, fix_roots: bool = True, fix_leaves: bool = True) -> BranchNetwork:
        """Apply Laplacian smoothing to node positions."""
        ...

    def subdivide(self) -> BranchNetwork:
        """Insert midpoints into every segment."""
        ...
```

## Execution Order
Typically, these are applied in the following order:
1.  **Decimate**: Clean up the raw output.
2.  **Subdivide**: Increase resolution.
3.  **Relocate**: Smooth the resulting high-resolution network.
4.  **Relocate** (again): Multiple passes of smoothing may be used.
