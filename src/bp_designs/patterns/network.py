"""Branch network data structure for semantic and vectorized operations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import KDTree

from bp_designs.core.geometry import Polyline
from bp_designs.core.pattern import Pattern


@dataclass
class BranchNetwork(Pattern):
    """Semantic structure for branching patterns.



    Internal Structure (Semantic):


        All arrays have length N (number of nodes).


        - positions: (N, 2) - Node positions


        - parents: (N,) - Parent indices (-1 for roots)


        - depths: (N,) - Hierarchy depth from root


        - branch_ids: (N,) - Which branch each node belongs to


        - timestamps: (N,) - Growth order (iteration when added)



    Pattern Interface:


        Exposes tree structure as queryable spatial fields:


        - 'distance': Distance to nearest node


        - 'depth': Interpolated hierarchy depth


        - 'branch_id': ID of nearest branch


        - 'density': Exponential falloff from branches


        - 'direction': Growth direction (returns (N, 2))
    """

    node_ids: np.ndarray  # (N,) - Node IDs

    positions: np.ndarray  # (N, 2) - Node positions

    parents: np.ndarray  # (N,) - Parent indices (-1 for roots)

    timestamps: np.ndarray  # (N,) - Growth order (iteration when added)

    # Field query configuration

    density_falloff: float = 10.0  # Exponential falloff distance for density field

    # Cached spatial index for fast field queries (lazy initialization)

    _kdtree: KDTree | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        """Validate structure."""
        n = len(self.positions)

        assert self.parents.shape == (n,), "parents must be (N,)"

        assert self.timestamps.shape == (n,), "timestamps must be (N,)"

        assert self.positions.shape == (n, 2), "positions must be (N, 2)"

    # ============================================================================

    # Pattern Interface Implementation

    # ============================================================================

    @property
    def kdtree(self) -> KDTree:
        """Lazy-initialized spatial index for fast field queries.



        KDTree enables O(log N) nearest neighbor queries instead of O(N).


        Built only when first field query is made.
        """

        if self._kdtree is None:
            self._kdtree = KDTree(self.positions)

        return self._kdtree

    def sample_field(self, points: np.ndarray, channel: str) -> np.ndarray:
        """Sample branch network as a field (vectorized).



        Uses cached KDTree for O(log N) nearest neighbor queries.


        All operations are vectorized (no Python loops).



        Args:


            points: (N, 2) array of query positions


            channel: One of: 'distance', 'depth', 'branch_id', 'density', 'direction'



        Returns:


            (N,) array for scalar channels, (N, 2) for 'direction'



        Raises:


            ValueError: If channel is unknown
        """

        if channel == "distance":
            # Distance to nearest branch node

            distances, _ = self.kdtree.query(points)
            return distances

        elif channel == "depth":
            # Interpolate hierarchy depth from nearest node

            _, indices = self.kdtree.query(points)

            return self.depths[indices].astype(float)

        elif channel == "branch_id":
            # Which branch is nearest

            _, indices = self.kdtree.query(points)

            return self.branch_ids[indices].astype(float)

        elif channel == "density":
            # Branch density with exponential falloff

            distances, _ = self.kdtree.query(points)

            return np.exp(-distances / self.density_falloff)

        elif channel == "direction":
            # Growth direction at nearest point (unit vector)

            _, indices = self.kdtree.query(points)

            # Compute direction from parent to node

            parent_idx = self.parents[indices]

            # Handle roots (parent = -1)

            mask = parent_idx >= 0

            directions = np.zeros((len(points), 2))

            if mask.any():
                directions[mask] = self.positions[indices[mask]] - self.positions[parent_idx[mask]]

                # Normalize

                norms = np.linalg.norm(directions[mask], axis=1, keepdims=True)

                norms = np.maximum(norms, 1e-10)  # Avoid division by zero

                directions[mask] /= norms

            return directions

        else:
            raise ValueError(
                f"Unknown channel '{channel}'. Available: {list(self.available_channels().keys())}"
            )

    def available_channels(self) -> dict[str, str]:
        """Return available field channels with descriptions."""

        return {
            "distance": "Distance to nearest branch node",
            "depth": "Hierarchy depth from root (interpolated from nearest node)",
            "branch_id": "ID of nearest branch",
            "density": f"Branch density (exp(-d/{self.density_falloff}))",
            "direction": "Growth direction (unit vector, returns (N,2) array)",
        }

    def to_geometry(self) -> Polyline:
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

        return Polyline(polylines=branches)

    def bounds(self) -> tuple[float, float, float, float]:
        """Compute tight bounding box around all nodes.



        Returns:


            (xmin, ymin, xmax, ymax)
        """

        xmin, ymin = self.positions.min(axis=0)

        xmax, ymax = self.positions.max(axis=0)

        return (float(xmin), float(ymin), float(xmax), float(ymax))

    # ============================================================================

    # Semantic Network Properties

    # ============================================================================

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

    def get_latest_nodes(self) -> BranchNetwork:
        """Get subset of network up to and including a specific timestamp.



        Args:


            timestamp: Maximum timestamp to include



        Returns:


            New BranchNetwork containing only nodes with timestamps <= given timestamp
        """

        latest_timestamp = np.max(self.timestamps)

        mask = self.timestamps == latest_timestamp

        return self.get_nodes(mask)

    def get_nodes(self, selection) -> BranchNetwork:
        return BranchNetwork(
            node_ids=self.node_ids[selection],
            positions=self.positions[selection],
            parents=self.parents[selection],
            timestamps=self.timestamps[selection],
            density_falloff=self.density_falloff,
        )

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

    # @classmethod

    # def from_node_list(cls, nodes: list[dict], compute_branches: bool = True) -> BranchNetwork:

    #     """Convert list-of-dicts representation to vectorized structure.

    #     Useful for converting from legacy implementation or simple builders.

    #     Args:

    #         nodes: List of {"pos": np.array, "parent": int, "timestamp": int}

    #         compute_branches: If True, compute branch IDs by tracing to leaves

    #     Returns:

    #         BranchNetwork instance

    #     """

    #     n = len(nodes)

    #     # Extract arrays

    #     positions = np.array([node["pos"] for node in nodes])

    #     parents = np.array([node.get("parent", -1) for node in nodes], dtype=int)

    #     timestamps = np.array([node.get("timestamp", i) for i, node in enumerate(nodes)])

    #     # Compute depths

    #     depths = np.zeros(n, dtype=int)

    #     for i in range(n):

    #         depth = 0

    #         current = i

    #         while parents[current] != -1:

    #             depth += 1

    #             current = parents[current]

    #             if depth > n:  # Cycle detection

    #                 raise ValueError(f"Cycle detected at node {i}")

    #         depths[i] = depth

    #     # Compute branch IDs if requested

    #     if compute_branches:

    #         branch_ids = cls._compute_branch_ids(parents)

    #     else:

    #         branch_ids = np.zeros(n, dtype=int)

    #     return cls(

    #         positions=positions,

    #         parents=parents,

    #         depths=depths,

    #         branch_ids=branch_ids,

    #         timestamps=timestamps,

    #     )

    @staticmethod
    def _compute_depths(parents) -> np.ndarray:
        n = len(parents)

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
        return depths

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
