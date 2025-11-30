"""Branch network data structure for semantic and vectorized operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bp_designs.geometry import Geometry


@dataclass
class BranchNetwork:
    """Vectorized branching structure preserving semantic information.

    This structure enables:
    - Fast vectorized operations (no Python loops)
    - Semantic queries (hierarchy, branch identity)
    - Advanced processing (tapering, coloring, analysis)

    All arrays have length N (number of nodes).
    """

    positions: np.ndarray  # (N, 2) - Node positions
    parents: np.ndarray  # (N,) - Parent indices (-1 for roots)
    depths: np.ndarray  # (N,) - Hierarchy depth from root
    branch_ids: np.ndarray  # (N,) - Which branch each node belongs to
    timestamps: np.ndarray  # (N,) - Growth order (iteration when added)

    def __post_init__(self):
        """Validate structure."""
        n = len(self.positions)
        assert self.parents.shape == (n,), "parents must be (N,)"
        assert self.depths.shape == (n,), "depths must be (N,)"
        assert self.branch_ids.shape == (n,), "branch_ids must be (N,)"
        assert self.timestamps.shape == (n,), "timestamps must be (N,)"
        assert self.positions.shape == (n, 2), "positions must be (N, 2)"

    @property
    def num_nodes(self) -> int:
        """Number of nodes in network."""
        return len(self.positions)

    @property
    def num_branches(self) -> int:
        """Number of unique branches."""
        return len(np.unique(self.branch_ids))

    @property
    def max_depth(self) -> int:
        """Maximum hierarchy depth."""
        return int(np.max(self.depths))

    def get_branch(self, branch_id: int) -> np.ndarray:
        """Extract single branch as ordered polyline.

        Args:
            branch_id: Branch identifier

        Returns:
            (M, 2) array of positions along branch (root → leaf order)
        """
        mask = self.branch_ids == branch_id
        branch_positions = self.positions[mask]
        branch_timestamps = self.timestamps[mask]

        # Sort by timestamp to get root → leaf order
        order = np.argsort(branch_timestamps)
        return branch_positions[order]

    def get_leaves(self) -> np.ndarray:
        """Get indices of leaf nodes (nodes with no children).

        Returns:
            Array of leaf node indices
        """
        # A node is a leaf if it's not a parent of any other node
        is_parent = np.isin(np.arange(self.num_nodes), self.parents)
        return np.where(~is_parent)[0]

    def get_roots(self) -> np.ndarray:
        """Get indices of root nodes (nodes with no parent).

        Returns:
            Array of root node indices
        """
        return np.where(self.parents == -1)[0]

    def taper_weights(self, base_width: float = 1.0, taper_rate: float = 0.8) -> np.ndarray:
        """Compute stroke widths based on hierarchy depth.

        Vectorized: width = base_width * taper_rate^depth

        Args:
            base_width: Width at root
            taper_rate: Multiplicative factor per level (0-1)

        Returns:
            (N,) array of widths for each node
        """
        return base_width * (taper_rate**self.depths)

    def to_geometry(self) -> Geometry:
        """Convert to polyline representation for export.

        Extracts each branch as a complete path from leaf to root.
        Nodes can appear in multiple branches (shared trunk segments).

        Returns:
            List of polylines (each is Mx2 array)
        """
        leaves = self.get_leaves()
        branches = []

        for leaf_idx in leaves:
            # Trace from leaf to root
            path = []
            current = leaf_idx
            while current != -1:
                path.append(self.positions[current])
                current = self.parents[current]

            if len(path) >= 2:
                # Reverse to get root → leaf order
                branches.append(np.array(path[::-1]))

        return branches

    @classmethod
    def from_node_list(cls, nodes: list[dict], compute_branches: bool = True) -> BranchNetwork:
        """Convert list-of-dicts representation to vectorized structure.

        Useful for converting from legacy implementation or simple builders.

        Args:
            nodes: List of {"pos": np.array, "parent": int, "timestamp": int}
            compute_branches: If True, compute branch IDs by tracing to leaves

        Returns:
            BranchNetwork instance
        """
        n = len(nodes)

        # Extract arrays
        positions = np.array([node["pos"] for node in nodes])
        parents = np.array([node.get("parent", -1) for node in nodes], dtype=int)
        timestamps = np.array([node.get("timestamp", i) for i, node in enumerate(nodes)])

        # Compute depths
        depths = np.zeros(n, dtype=int)
        for i in range(n):
            depth = 0
            current = i
            while parents[current] != -1:
                depth += 1
                current = parents[current]
                if depth > n:  # Cycle detection
                    raise ValueError(f"Cycle detected at node {i}")
            depths[i] = depth

        # Compute branch IDs if requested
        if compute_branches:
            branch_ids = cls._compute_branch_ids(parents)
        else:
            branch_ids = np.zeros(n, dtype=int)

        return cls(
            positions=positions,
            parents=parents,
            depths=depths,
            branch_ids=branch_ids,
            timestamps=timestamps,
        )

    @staticmethod
    def _compute_branch_ids(parents: np.ndarray) -> np.ndarray:
        """Assign branch ID to each node by tracing from leaves to roots.

        Each path from leaf to root gets a unique branch ID.

        Args:
            parents: (N,) array of parent indices

        Returns:
            (N,) array of branch IDs
        """
        n = len(parents)
        branch_ids = np.full(n, -1, dtype=int)

        # Find leaves (nodes that are not parents)
        is_parent = np.isin(np.arange(n), parents)
        leaves = np.where(~is_parent)[0]

        # Trace from each leaf to root, assigning branch ID
        next_branch_id = 0
        for leaf in leaves:
            current = leaf
            while current != -1 and branch_ids[current] == -1:
                branch_ids[current] = next_branch_id
                current = parents[current]
            next_branch_id += 1

        return branch_ids
