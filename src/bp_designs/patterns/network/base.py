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
    pattern_bounds: tuple[float, float, float, float] | None = None  # (xmin, ymin, xmax, ymax)
    thickness: np.ndarray | None = None  # (N,) - Optional stored thickness values
    colors: np.ndarray | None = None  # (N,) - Optional stored color values (hex strings)
    organs: dict[int, OrganPattern] | None = None  # node_id -> Organ mapping
    organ_template: OrganPattern | None = None  # Template for automatic organ attachment
    organ_distribution: str | Any = "terminal"  # Strategy for automatic organ attachment

    def __post_init__(self):
        """Validate structure and attach organs if template provided."""
        n = len(self.positions)
        assert self.parents.shape == (n,), f"parents must be ({n},), got {self.parents.shape}"
        assert self.timestamps.shape == (n,), f"timestamps must be ({n},), got {self.timestamps.shape}"
        assert self.positions.shape == (n, 2), f"positions must be ({n}, 2), got {self.positions.shape}"
        if self.thickness is not None:
            assert self.thickness.shape == (n,), f"thickness must be ({n},), got {self.thickness.shape}"

        # Automatically attach organs if template is provided and organs not already set
        if self.organ_template is not None and self.organs is None:
            self.attach_organs(self.organ_template, self.organ_distribution)

    @cached_property
    def depths(self) -> np.ndarray:
        """Hierarchy depth from root for each node."""
        n = len(self.parents)
        depths = np.zeros(n, dtype=int)
        for i in range(n):
            depth = 0
            current = i
            while self.parents[current] != -1:
                depth += 1
                current = self.parents[current]
                if depth > n:  # Cycle detection
                    raise ValueError(f"Cycle detected at node {i}")
            depths[i] = depth
        return depths

    @cached_property
    def branch_ids(self) -> np.ndarray:
        """Assign branch ID to each node by tracing from leaves to roots."""
        n = len(self.parents)
        branch_ids = np.full(n, -1, dtype=int)
        leaves = self.get_leaves()
        next_branch_id = 0
        for leaf in leaves:
            current = leaf
            while current != -1 and branch_ids[current] == -1:
                branch_ids[current] = next_branch_id
                current = self.parents[current]
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
        is_parent = np.isin(np.arange(self.num_nodes), self.parents)
        return np.where(~is_parent)[0]

    def get_roots(self) -> np.ndarray:
        return np.where(self.parents == -1)[0]

    def get_nodes(self, selection) -> BranchNetwork:
        return BranchNetwork(
            node_ids=self.node_ids[selection],
            positions=self.positions[selection],
            parents=self.parents[selection],
            timestamps=self.timestamps[selection],
            pattern_bounds=self.pattern_bounds,
            organ_template=self.organ_template,
            organ_distribution=self.organ_distribution,
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
        for _ in range(iterations):
            neighbor_sum = np.zeros_like(new_positions)
            neighbor_count = np.zeros(self.num_nodes)
            np.add.at(neighbor_sum, child_indices, new_positions[parent_indices])
            np.add.at(neighbor_count, child_indices, 1)
            np.add.at(neighbor_sum, parent_indices, new_positions[child_indices])
            np.add.at(neighbor_count, parent_indices, 1)
            mask = neighbor_count > 0
            avg_positions = np.zeros_like(new_positions)
            avg_positions[mask] = neighbor_sum[mask] / neighbor_count[mask, None]
            move_mask = mask & to_move
            new_positions[move_mask] = (1 - alpha) * new_positions[move_mask] + alpha * avg_positions[
                move_mask
            ]
        return BranchNetwork(
            node_ids=self.node_ids.copy(),
            positions=new_positions,
            parents=self.parents.copy(),
            timestamps=self.timestamps.copy(),
            pattern_bounds=self.pattern_bounds,
            thickness=self.thickness.copy() if self.thickness is not None else None,
            organ_template=self.organ_template,
            organ_distribution=self.organ_distribution,
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
        new_thickness = None
        if self.thickness is not None:
            new_thickness_vals = (self.thickness[parent_indices] + self.thickness[child_indices]) / 2.0
            new_thickness = np.concatenate([self.thickness, new_thickness_vals])
        return BranchNetwork(
            node_ids=np.concatenate([self.node_ids, new_node_ids]),
            positions=np.concatenate([self.positions, midpoints]),
            parents=np.concatenate([updated_parents, new_parents]),
            timestamps=np.concatenate([self.timestamps, new_timestamps]),
            pattern_bounds=self.pattern_bounds,
            thickness=new_thickness,
            organ_template=self.organ_template,
            organ_distribution=self.organ_distribution,
        )

    def decimate(self, min_distance: float = 1.0) -> BranchNetwork:
        if self.num_nodes <= 1:
            return self
        has_parent = self.parents != -1
        child_indices = np.where(has_parent)[0]
        parent_ids = self.parents[has_parent]
        id_to_idx = {node_id: i for i, node_id in enumerate(self.node_ids)}
        parent_indices = np.array([id_to_idx[pid] for pid in parent_ids])
        dists = np.linalg.norm(self.positions[child_indices] - self.positions[parent_indices], axis=1)
        to_remove_mask = np.zeros(self.num_nodes, dtype=bool)
        to_remove_mask[child_indices[dists < min_distance]] = True
        to_remove_mask[self.get_roots()] = False
        if not np.any(to_remove_mask):
            return self
        new_parents = self.parents.copy()
        for i in range(self.num_nodes):
            if self.parents[i] == -1:
                continue
            curr_parent_id = self.parents[i]
            while curr_parent_id != -1:
                curr_parent_idx = id_to_idx[curr_parent_id]
                if not to_remove_mask[curr_parent_idx]:
                    break
                curr_parent_id = self.parents[curr_parent_idx]
            new_parents[i] = curr_parent_id
        keep_mask = ~to_remove_mask
        return BranchNetwork(
            node_ids=self.node_ids[keep_mask],
            positions=self.positions[keep_mask],
            parents=new_parents[keep_mask],
            timestamps=self.timestamps[keep_mask],
            pattern_bounds=self.pattern_bounds,
            thickness=self.thickness[keep_mask] if self.thickness is not None else None,
            organ_template=self.organ_template,
            organ_distribution=self.organ_distribution,
        )

    # ============================================================================
    # Strategy Delegation
    # ============================================================================

    def attach_organs(self, organ_template: OrganPattern, distribution: str | Any = "terminal", **kwargs):
        from bp_designs.patterns.network.distribution import OrganDistributionStrategy

        if self.organs is None:
            self.organs = {}
        strategy = OrganDistributionStrategy.from_name(distribution, **kwargs.pop("distribution_params", {}))
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
