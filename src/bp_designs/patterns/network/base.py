"""Core BranchNetwork data structure and refinement logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from bp_designs.core.geometry import Canvas, Polyline
from bp_designs.core.pattern import Pattern

if TYPE_CHECKING:
    from bp_designs.core.geometry import Polygon
    from bp_designs.core.renderer import RenderingContext
    from bp_designs.core.scene import Layer
    from bp_designs.patterns.organs import OrganPattern


@dataclass(kw_only=True)
class BranchNetwork(Pattern):
    """Semantic structure for branching patterns.

    Internal Structure (Semantic):
        All arrays have length N (number of nodes).
        - positions: (N, 2) - Node positions
        - parents: (N,) - Parent indices (-1 for roots)
        - depths: (N,) - Hierarchy depth from root (cached property)
        - branch_ids: (N,) - Which branch each node belongs to (cached property)
        - timestamps: (N,) - Growth order (iteration when added)
    """

    node_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))
    positions: np.ndarray = field(default_factory=lambda: np.array([], dtype=float).reshape(0, 2))
    parents: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))
    velocities: np.ndarray | None = None  # (N, 2) - Optional stored growth vectors
    pattern_bounds: tuple[float, float, float, float] | None = None  # (xmin, ymin, xmax, ymax)
    thickness: np.ndarray | None = None  # (N,) - Optional stored thickness values
    colors: np.ndarray | None = None  # (N,) - Optional stored color values (hex strings)
    organs: dict[int, OrganPattern] | None = None  # node_id -> Organ mapping
    organ_template: OrganPattern | None = None  # Template for automatic organ attachment
    organ_distribution: str | Any = "terminal"  # Strategy for automatic organ attachment

    def __post_init__(self):
        """Validate structure."""
        n = len(self.positions)
        assert self.parents.shape == (n,), f"parents must be ({n},), got {self.parents.shape}"
        assert self.timestamps.shape == (n,), f"timestamps must be ({n},), got {self.timestamps.shape}"
        assert self.positions.shape == (n, 2), f"positions must be ({n}, 2), got {self.positions.shape}"
        if self.velocities is not None:
            assert self.velocities.shape == (n, 2), f"velocities must be ({n}, 2), got {self.velocities.shape}"
        if self.thickness is not None:
            assert self.thickness.shape == (n,), f"thickness must be ({n},), got {self.thickness.shape}"

    def replace(self, **changes) -> BranchNetwork:
        """Create a new instance with updated fields, preserving others.

        This is a semantic-aware version of dataclasses.replace that ensures
        cached properties are cleared and all state is carried over.
        """
        from dataclasses import replace

        return replace(self, **changes)

    @cached_property
    def depths(self) -> np.ndarray:
        """Hierarchy depth from root for each node."""
        n = len(self.parents)
        depths = np.zeros(n, dtype=int)
        id_to_idx = {node_id: i for i, node_id in enumerate(self.node_ids)}
        for i in range(n):
            depth = 0
            curr_idx = i
            while self.parents[curr_idx] != -1:
                depth += 1
                parent_id = self.parents[curr_idx]
                curr_idx = id_to_idx[parent_id]
                if depth > n:  # Cycle detection
                    raise ValueError(f"Cycle detected at node {i}")
            depths[i] = depth
        return depths

    @cached_property
    def branch_ids(self) -> np.ndarray:
        """Assign branch ID to each node by tracing from leaves to roots."""
        n = len(self.parents)
        branch_ids = np.full(n, -1, dtype=int)
        id_to_idx = {node_id: i for i, node_id in enumerate(self.node_ids)}
        leaves = self.get_leaves()
        next_branch_id = 0
        for leaf_idx in leaves:
            curr_idx = leaf_idx
            while curr_idx != -1 and branch_ids[curr_idx] == -1:
                branch_ids[curr_idx] = next_branch_id
                parent_id = self.parents[curr_idx]
                if parent_id == -1:
                    curr_idx = -1
                else:
                    curr_idx = id_to_idx[parent_id]
            next_branch_id += 1
        return branch_ids

    # ============================================================================
    # Pattern Interface Implementation
    # ============================================================================

    def to_geometry(self, canvas: Canvas | None = None) -> Polyline:
        from bp_designs.patterns.network.renderer import NetworkRenderer

        return NetworkRenderer(self).to_geometry(canvas)

    def render(self, context: RenderingContext, **kwargs):
        from bp_designs.patterns.network.renderer import NetworkRenderer

        NetworkRenderer(self).render(context, **kwargs)

    def to_layers(self) -> list[Layer]:
        """Decompose network into branches and organs layers."""
        from bp_designs.core.scene import Layer

        layers = [
            Layer(name="branches", pattern=self, params={"render_organs": False}),
        ]
        if self.organs:
            layers.append(Layer(name="organs", pattern=self, params={"render_branches": False}))
        return layers

    def to_polygon(self, **kwargs) -> list[Polygon]:
        from bp_designs.patterns.network.renderer import NetworkRenderer

        return NetworkRenderer(self).to_polygon(**kwargs)

    def bounds(self) -> tuple[float, float, float, float]:
        """Compute tight bounding box around all nodes."""
        if len(self.positions) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        xmin, ymin = self.positions.min(axis=0)
        xmax, ymax = self.positions.max(axis=0)
        return (float(xmin), float(ymin), float(xmax), float(ymax))

    # ============================================================================
    # Semantic Network Properties
    # ============================================================================

    @property
    def num_nodes(self) -> int:
        return len(self.positions)

    @property
    def num_branches(self) -> int:
        return len(np.unique(self.branch_ids))

    @property
    def max_depth(self) -> int:
        return int(np.max(self.depths)) if len(self.depths) > 0 else 0

    def get_branch(self, branch_id: int) -> np.ndarray:
        mask = self.branch_ids == branch_id
        branch_positions = self.positions[mask]
        branch_timestamps = self.timestamps[mask]
        order = np.argsort(branch_timestamps)
        return branch_positions[order]

    def get_leaves(self) -> np.ndarray:
        is_parent = np.isin(self.node_ids, self.parents)
        return np.where(~is_parent)[0]

    def get_roots(self) -> np.ndarray:
        return np.where(self.parents == -1)[0]

    def get_nodes(self, selection) -> BranchNetwork:
        return BranchNetwork(
            node_ids=self.node_ids[selection],
            positions=self.positions[selection],
            parents=self.parents[selection],
            timestamps=self.timestamps[selection],
            canvas=self.canvas,
            pattern_bounds=self.pattern_bounds,
            organ_template=self.organ_template,
            organ_distribution=self.organ_distribution,
            velocities=self.velocities[selection] if self.velocities is not None else None,
        )

    # ============================================================================
    # Refinement Logic
    # ============================================================================

    def relocate(
        self,
        alpha: float = 0.5,
        iterations: int = 1,
        fix_roots: bool = True,
        fix_leaves: bool = True,
    ) -> BranchNetwork:
        """Relocate nodes toward their parents to reduce branching angles.

        This implements the basal relocation described in Runions et al. (2007).
        """
        if self.num_nodes <= 1:
            return self

        new_positions = self.positions.copy()
        to_move = np.ones(self.num_nodes, dtype=bool)
        if fix_roots:
            to_move[self.get_roots()] = False
        if fix_leaves:
            to_move[self.get_leaves()] = False

        id_to_idx = {node_id: i for i, node_id in enumerate(self.node_ids)}
        has_parent = self.parents != -1
        child_indices = np.where(has_parent)[0]
        parent_ids = self.parents[has_parent]
        parent_indices = np.array([id_to_idx[pid] for pid in parent_ids])

        move_mask = has_parent & to_move

        for _ in range(iterations):
            # Move each node toward its parent
            target_positions = new_positions.copy()
            target_positions[child_indices] = new_positions[parent_indices]

            new_positions[move_mask] = (1 - alpha) * new_positions[move_mask] + alpha * target_positions[
                move_mask
            ]

        return self.replace(
            positions=new_positions,
            thickness=self.thickness.copy() if self.thickness is not None else None,
        )

    def subdivide(self) -> BranchNetwork:
        has_parent = self.parents != -1
        child_indices = np.where(has_parent)[0]
        if len(child_indices) == 0:
            return self
        id_to_idx = {node_id: i for i, node_id in enumerate(self.node_ids)}
        parent_ids = self.parents[has_parent]
        parent_indices = np.array([id_to_idx[pid] for pid in parent_ids])
        midpoints = (self.positions[parent_indices] + self.positions[child_indices]) / 2.0
        max_id = self.node_ids.max()
        num_new_nodes = len(child_indices)
        new_node_ids = np.arange(max_id + 1, max_id + 1 + num_new_nodes)
        new_timestamps = (
            self.timestamps[parent_indices].astype(float) + self.timestamps[child_indices].astype(float)
        ) / 2.0
        new_timestamps = new_timestamps.astype(np.int16)
        updated_parents = self.parents.copy()
        updated_parents[child_indices] = new_node_ids
        new_parents = parent_ids
        return self.replace(
            node_ids=np.concatenate([self.node_ids, new_node_ids]),
            positions=np.concatenate([self.positions, midpoints]),
            parents=np.concatenate([updated_parents, new_parents]),
            timestamps=np.concatenate([self.timestamps, new_timestamps]),
            thickness=None,
            colors=None,
            velocities=None,
        )

    def decimate(self, min_distance: float = 1.0) -> BranchNetwork:
        """Remove nodes that are too close to their parents.

        Uses an iterative topological pass to allow segments to accumulate length,
        avoiding the 'all or nothing' issue with fixed-length growth.
        """
        if self.num_nodes <= 1:
            return self

        # Sort nodes by depth to ensure we process parents before children
        depths = self.depths
        order = np.argsort(depths)

        id_to_idx = {node_id: i for i, node_id in enumerate(self.node_ids)}
        new_parents = self.parents.copy()
        to_remove_mask = np.zeros(self.num_nodes, dtype=bool)

        for idx in order:
            parent_id = new_parents[idx]
            if parent_id == -1:
                continue

            parent_idx = id_to_idx[parent_id]
            dist = np.linalg.norm(self.positions[idx] - self.positions[parent_idx])

            if dist < min_distance:
                to_remove_mask[idx] = True
                # Re-parent children of this node to its parent
                node_id = self.node_ids[idx]
                children_mask = new_parents == node_id
                new_parents[children_mask] = parent_id

        if not np.any(to_remove_mask):
            return self

        keep_mask = ~to_remove_mask

        # Map old parent IDs to their new values (which might be -1 or other kept IDs)
        # We need to ensure that the final parents array only contains IDs that are in the kept node_ids.
        # The iterative re-parenting above already ensures this because we only re-parent to the parent
        # of the removed node, and we process in topological order.
        # However, we must convert the parent IDs to the new indices if we were using indices,
        # but our BranchNetwork uses IDs for parents, so we just need to filter.

        # Filter organs to keep only those on remaining nodes
        new_organs = None
        if self.organs:
            new_organs = {
                node_id: organ
                for node_id, organ in self.organs.items()
                if node_id in self.node_ids[keep_mask]
            }

        return self.replace(
            node_ids=self.node_ids[keep_mask],
            positions=self.positions[keep_mask],
            parents=new_parents[keep_mask],
            timestamps=self.timestamps[keep_mask],
            thickness=None,
            colors=None,
            velocities=self.velocities[keep_mask] if self.velocities is not None else None,
            organs=new_organs,
        )

    # ============================================================================
    # Strategy Delegation
    # ============================================================================

    def attach_organs(self, organ_template: OrganPattern, distribution: str | Any = "terminal", **kwargs):
        from bp_designs.patterns.network.distribution import OrganDistributionStrategy

        if self.organs is None:
            self.organs = {}

        # If distribution is already a strategy, use it. Otherwise, create from name.
        if isinstance(distribution, OrganDistributionStrategy):
            strategy = distribution
        else:
            dist_params = kwargs.pop("distribution_params", {})
            strategy = OrganDistributionStrategy.from_name(distribution, **dist_params)

        new_organs = strategy.generate_organs(self, organ_template, **kwargs)
        for node_id, organ in new_organs:
            self.organs[node_id] = organ

    def __str__(self) -> str:
        return f"BranchNetwork(N={self.num_nodes})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BranchNetwork):
            return False
        return (
            np.array_equal(self.node_ids, other.node_ids)
            and np.array_equal(self.positions, other.positions)
            and np.array_equal(self.parents, other.parents)
            and np.array_equal(self.timestamps, other.timestamps)
            and np.array_equal(self.velocities, other.velocities)
        )

    def __hash__(self) -> int:
        def array_hash(arr):
            if arr.size == 0:
                return hash(())
            return hash(tuple(tuple(map(float, point)) for point in arr.reshape(-1, arr.shape[-1])))

        return hash(
            (
                array_hash(self.node_ids),
                array_hash(self.positions),
                array_hash(self.parents),
                array_hash(self.timestamps),
            )
        )
