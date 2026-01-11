"""Branch network data structure for semantic and vectorized operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import svgwrite

from bp_designs.core.geometry import Canvas, Polyline
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

    pattern_bounds: tuple[float, float, float, float] | None = None  # (xmin, ymin, xmax, ymax) for framing

    thickness: np.ndarray | None = None  # (N,) - Optional stored thickness values



    def __post_init__(self):
        """Validate structure."""
        n = len(self.positions)

        assert self.parents.shape == (n,), "parents must be (N,)"

        assert self.timestamps.shape == (n,), "timestamps must be (N,)"

        assert self.positions.shape == (n, 2), "positions must be (N, 2)"

        if self.thickness is not None:
            assert self.thickness.shape == (n,), "thickness must be (N,)"

    # ============================================================================

    # Pattern Interface Implementation

    # ============================================================================



    def to_geometry(self, canvas: Canvas | None = None) -> Polyline:
        """Convert to polyline representation for export.

        Extracts each branch as a complete path from leaf to root.
        Nodes can appear in multiple branches (shared trunk segments).

        If canvas is provided, the network is scaled to fit the canvas bounds
        relative to its original pattern_bounds.

        Returns:
            List of polylines (each is Mx2 array)
        """
        positions = self.positions

        if canvas is not None and self.pattern_bounds is not None:
            # Scale positions to fit new canvas
            orig_xmin, orig_ymin, orig_xmax, orig_ymax = self.pattern_bounds
            orig_width = orig_xmax - orig_xmin
            orig_height = orig_ymax - orig_ymin

            new_xmin, new_ymin, new_xmax, new_ymax = canvas.bounds()
            new_width = new_xmax - new_xmin
            new_height = new_ymax - new_ymin

            if orig_width > 0 and orig_height > 0:
                positions = positions.copy()
                # Normalize to 0-1
                positions[:, 0] = (positions[:, 0] - orig_xmin) / orig_width
                positions[:, 1] = (positions[:, 1] - orig_ymin) / orig_height
                # Scale to new canvas
                positions[:, 0] = new_xmin + positions[:, 0] * new_width
                positions[:, 1] = new_ymin + positions[:, 1] * new_height

        leaves = self.get_leaves()
        branches = []

        for leaf_idx in leaves:
            # Trace from leaf to root
            path = []
            current = leaf_idx
            while current != -1:
                path.append(positions[current])
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

    def _compute_thickness_values(
        self,
        thickness: str = 'descendant',
        min_thickness: float = 0.5,
        max_thickness: float = 5.0,
        taper_power: float = 0.5,
        thickness_mode: str = 'all_nodes',
    ) -> np.ndarray:
        """Compute thickness for each node based on strategy.
        Args:
            thickness: Strategy name ('timestamp', 'hierarchy', 'descendant')
            min_thickness: Minimum thickness value
            max_thickness: Maximum thickness value
            taper_power: Power law exponent for descendant mode
            thickness_mode: Mode for descendant strategy ('all_nodes', 'leaves_only')

        Returns:
            (N,) array of thickness values for each node
        """
        if thickness == 'timestamp':
            strategy = TimestampThickness(min_thickness, max_thickness)
        elif thickness == 'hierarchy':
            strategy = HierarchyThickness(min_thickness, max_thickness)
        elif thickness == 'descendant':
            strategy = DescendantThickness(min_thickness, max_thickness, taper_power, thickness_mode)
        else:
            raise ValueError(f"Unknown thickness strategy: {thickness}")

        thickness_values = strategy.compute_thickness(self)
        self.thickness = thickness_values
        return thickness_values

    def to_svg(
        self,
        thickness: str = 'descendant',
        min_thickness: float = 0.5,
        max_thickness: float = 5.0,
        taper_power: float = 0.5,
        thickness_mode: str = 'all_nodes',
        taper_style: str = 'smooth',  # 'smooth' or 'blocky'
        color: str = 'black',
        stroke_linecap: str = 'round',
        stroke_linejoin: str = 'round',
        width: str | float = '100%',
        height: str | float = '100%',
        padding: float = 20,
        **kwargs
    ) -> str:
        """Render branch network to SVG with semantic thickness.

        Args:
            thickness: Thickness strategy ('timestamp', 'hierarchy', 'descendant')
            min_thickness: Minimum branch thickness
            max_thickness: Maximum branch thickness
            taper_power: Power law for descendant thickness
            thickness_mode: Mode for descendant strategy ('all_nodes', 'leaves_only')
            taper_style: 'smooth' for interpolated or 'blocky' for per-segment
            color: Stroke color
            stroke_linecap: SVG linecap style ('round', 'butt', 'square')
            stroke_linejoin: SVG linejoin style ('round', 'miter', 'bevel')
            width: SVG canvas width (default '100%' for responsive)
            height: SVG canvas height (default '100%' for responsive)
            padding: Padding around content
            **kwargs: Additional SVG attributes

        Returns:
            SVG string
        """


        # Compute thickness for all nodes
        # This returns an array of length N (one for each node)
        # Use stored thickness if available and matches current request, otherwise compute
        if self.thickness is not None and len(self.thickness) == len(self.node_ids):
            all_thickness = self.thickness
        else:
            all_thickness = self._compute_thickness_values(
                thickness, min_thickness, max_thickness, taper_power, thickness_mode
            )

        # Compute view box from bounds
        if self.pattern_bounds is not None:
            xmin, ymin, xmax, ymax = self.pattern_bounds
        else:
            xmin, ymin, xmax, ymax = self.bounds()

        # Add padding
        xmin -= padding
        ymin -= padding
        xmax += padding
        ymax += padding

        view_width = xmax - xmin
        view_height = ymax - ymin

        # Format size for svgwrite
        def format_size(s):
            if isinstance(s, str):
                return s
            return f'{s}px'

        # Create SVG
        dwg = svgwrite.Drawing(
            size=(format_size(width), format_size(height)),
            viewBox=f'{xmin} {ymin} {view_width} {view_height}',
            **kwargs
        )


        # Smooth tapering: use path with varying stroke-width via opacity hack
        # Actually, SVG doesn't support varying stroke width along a path natively
        # So we render each segment separately with interpolated thickness

        # Create a mapping from node_id to index for O(1) lookup
        id_to_idx = {node_id: i for i, node_id in enumerate(self.node_ids)}

        for i in range(len(self.node_ids)):
            parent_id = self.parents[i]
            if parent_id == -1:
                continue

            # Get parent index
            if parent_id not in id_to_idx:
                continue
            parent_idx = id_to_idx[parent_id]

            # Get positions
            start = self.positions[parent_idx]
            end = self.positions[i]

            # Get thickness at both ends
            t_start = all_thickness[parent_idx]
            t_end = all_thickness[i]

            if taper_style == 'smooth':
                # Calculate normal vector for trapezoid corners
                v = end - start
                length = np.linalg.norm(v)
                if length > 0:
                    # Normal vector (perpendicular to segment)
                    n = np.array([-v[1], v[0]]) / length

                    # Four corners of the trapezoid
                    p1 = start + n * (t_start / 2)
                    p2 = start - n * (t_start / 2)
                    p3 = end - n * (t_end / 2)
                    p4 = end + n * (t_end / 2)

                    # Draw trapezoid as polygon
                    poly = dwg.polygon(
                        points=[
                            (float(p1[0]), float(p1[1])),
                            (float(p2[0]), float(p2[1])),
                            (float(p3[0]), float(p3[1])),
                            (float(p4[0]), float(p4[1])),
                        ],
                        fill=color,
                        **kwargs
                    )
                    dwg.add(poly)

                    # Add circular joints to smooth out junctions
                    # We add a circle at the start and end of each segment
                    # This ensures branching points are rounded and gaps are filled
                    dwg.add(dwg.circle(
                        center=(float(start[0]), float(start[1])),
                        r=float(t_start / 2),
                        fill=color,
                        **kwargs
                    ))
                    dwg.add(dwg.circle(
                        center=(float(end[0]), float(end[1])),
                        r=float(t_end / 2),
                        fill=color,
                        **kwargs
                    ))
                continue

            elif taper_style == 'blocky':
                # Use the thickness of the node itself
                t = all_thickness[i]
                # Draw line segment
                line = dwg.line(
                    start=(float(start[0]), float(start[1])),
                    end=(float(end[0]), float(end[1])),
                    stroke=color,
                    stroke_width=t,
                    stroke_linecap=stroke_linecap,
                    stroke_linejoin=stroke_linejoin,
                )
                dwg.add(line)
            else:
                raise ValueError(f"Unknown taper_style: {taper_style}")


        return dwg.tostring()

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
            pattern_bounds=self.pattern_bounds
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

    def __str__(self) -> str:
        """Return human-readable string representation."""
        num_nodes = len(self.positions) if self.positions is not None else 0
        return f"BranchNetwork(N={num_nodes})"

    def __eq__(self, other: object) -> bool:
        """Equality based on data arrays."""
        if not isinstance(other, BranchNetwork):
            return False
        # Compare arrays using np.array_equal
        if not np.array_equal(self.node_ids, other.node_ids):
            return False
        if not np.array_equal(self.positions, other.positions):
            return False
        if not np.array_equal(self.parents, other.parents):
            return False
        if not np.array_equal(self.timestamps, other.timestamps):
            return False
        return True

    def __hash__(self) -> int:
        """Hash based on data arrays."""
        # Convert arrays to tuples for hashing
        def array_hash(arr):
            if arr.size == 0:
                return hash(())
            return hash(tuple(tuple(map(float, point)) for point in arr.reshape(-1, arr.shape[-1])))

        node_ids_hash = array_hash(self.node_ids)
        positions_hash = array_hash(self.positions)
        parents_hash = array_hash(self.parents)
        timestamps_hash = array_hash(self.timestamps)

        return hash((node_ids_hash, positions_hash, parents_hash, timestamps_hash))


class BranchThicknessStrategy:
    """Base class for computing branch thickness from network structure."""

    def compute_thickness(self, network: BranchNetwork) -> np.ndarray:
        """Compute thickness value for each edge in the network.

        Args:
            network: BranchNetwork with semantic information

        Returns:
            (N,) array of thickness values for each node (excluding root)
        """
        raise NotImplementedError


class TimestampThickness(BranchThicknessStrategy):
    """Thickness based on node age/timestamp - older branches are thicker."""

    def __init__(self, min_thickness: float = 0.5, max_thickness: float = 5.0):
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness

    def compute_thickness(self, network: BranchNetwork) -> np.ndarray:
        """Compute thickness based on inverse timestamp (older = thicker)."""
        node_timestamps = network.timestamps

        max_time = network.timestamps.max()
        # Invert so older nodes (lower timestamp) get higher values
        if max_time > 0:
            age_normalized = (max_time - node_timestamps) / max_time
        else:
            age_normalized = np.zeros_like(node_timestamps)

        thickness = self.min_thickness + age_normalized * (self.max_thickness - self.min_thickness)
        return thickness


class HierarchyThickness(BranchThicknessStrategy):
    """Thickness based on distance from root - deeper branches are thinner."""

    def __init__(self, min_thickness: float = 0.5, max_thickness: float = 5.0):
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness

    def compute_thickness(self, network: BranchNetwork) -> np.ndarray:
        """Compute thickness based on hierarchical depth from root."""
        # Compute depth for each node
        depths = np.zeros(len(network.node_ids), dtype=int)

        for i, parent_id in enumerate(network.parents):
            if parent_id == -1:  # Root
                depths[i] = 0
            else:
                parent_idx = np.where(network.node_ids == parent_id)[0][0]
                depths[i] = depths[parent_idx] + 1

        max_depth = depths.max() if len(depths) > 0 else 1

        # Normalize and invert (shallow = thick)
        if max_depth > 0:
            depth_normalized = 1.0 - (depths / max_depth)
        else:
            depth_normalized = np.ones_like(depths, dtype=float)

        thickness = self.min_thickness + depth_normalized * (self.max_thickness - self.min_thickness)
        return thickness


class DescendantThickness(BranchThicknessStrategy):
    """Thickness based on number of terminal descendants - flow-based approach."""

    def __init__(
        self,
        min_thickness: float = 0.5,
        max_thickness: float = 5.0,
        power: float = 0.5,
        mode: str = 'all_nodes',
    ):
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.power = power  # For tapering curve (< 1 = gentle taper, > 1 = aggressive taper)
        self.mode = mode

    def compute_thickness(self, network: BranchNetwork) -> np.ndarray:
        """Compute thickness based on downstream terminal count."""
        num_nodes = len(network.node_ids)

        # Count descendants for each node
        if self.mode == 'leaves_only':
            # Only terminal nodes (leaves) count as flow
            descendant_counts = np.zeros(num_nodes, dtype=int)
            leaves = network.get_leaves()
            descendant_counts[leaves] = 1
        else:
            # Every node counts as 1 unit of flow (original behavior)
            descendant_counts = np.ones(num_nodes, dtype=int)

        # Propagate counts up the tree (work backwards through timestamps)
        # This ensures we process children before parents
        sorted_indices = np.argsort(network.timestamps)[::-1]  # Newest to oldest

        # Create a mapping from node_id to index for O(1) lookup
        id_to_idx = {node_id: i for i, node_id in enumerate(network.node_ids)}

        for idx in sorted_indices:
            parent_id = network.parents[idx]
            if parent_id >= 0:  # Not root
                # Use the mapping for robust and fast lookup
                if parent_id in id_to_idx:
                    parent_idx = id_to_idx[parent_id]
                    descendant_counts[parent_idx] += descendant_counts[idx]

        # Debug: print some counts to see if they are propagating
        # print(f"Max descendant count: {descendant_counts.max()}")

        # Use the root's descendant count as the max for normalization
        # This ensures the trunk is always max_thickness
        max_count = descendant_counts.max() if len(descendant_counts) > 0 else 1

        # Normalize with power law for tapering
        if max_count > 0:
             # We want the root to be 1.0 and leaves to be close to 0.0
             count_normalized = (descendant_counts / max_count) ** self.power
        else:
            count_normalized = np.zeros_like(descendant_counts, dtype=float)

        thickness = self.min_thickness + count_normalized * (self.max_thickness - self.min_thickness)
        return thickness
