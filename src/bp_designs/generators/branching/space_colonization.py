"""Space Colonization algorithm - vectorized implementation with semantic preservation."""

from collections.abc import Callable

import numpy as np
from scipy.spatial import cKDTree

from bp_designs.core.generator import Generator
from bp_designs.patterns.network import BranchNetwork


def _compute_influences_vectorized(
    node_positions: np.ndarray, attractions: np.ndarray, attraction_dist: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute which nodes are influenced by which attractions (fully vectorized).

    For each attraction point, finds the nearest node within attraction_dist.
    Uses broadcasting for O(1) numpy operations instead of nested loops.

    Args:
        node_positions: (N, 2) node positions
        attractions: (M, 2) attraction point positions
        attraction_dist: Maximum influence distance

    Returns:
        Tuple of:
            influenced_nodes: Array of node indices that are influenced
            attraction_indices: Corresponding attraction indices
    """
    # Vectorized distance computation using broadcasting
    # (N, 1, 2) - (1, M, 2) -> (N, M, 2) -> (N, M)
    diff = node_positions[:, np.newaxis, :] - attractions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))

    # Find nearest node for each attraction (M,)
    nearest_nodes = np.argmin(dists, axis=0)
    nearest_dists = np.min(dists, axis=0)

    # Filter by distance threshold
    valid_mask = nearest_dists < attraction_dist
    influenced_nodes = nearest_nodes[valid_mask]
    attraction_indices = np.where(valid_mask)[0]

    return influenced_nodes, attraction_indices


class SpaceColonization(Generator):
    """Generate branching patterns using Space Colonization algorithm.

    Vectorized implementation preserving semantic information (hierarchy, branch IDs).

    This algorithm simulates growth toward "attraction points" to create
    natural-looking vein or root-like structures.
    """

    def __init__(
        self,
        seed: int = 0,
        num_attractions: int = 500,
        attraction_distance: float = 50.0,
        kill_distance: float = 5.0,
        segment_length: float = 2.0,
        width: float = 100.0,
        height: float = 100.0,
        root_position: tuple[float, float] | None = None,
    ):
        """Initialize Space Colonization generator.

        Args:
            seed: Random seed for determinism
            num_attractions: Number of attraction points (growth targets)
            attraction_distance: Max distance for attraction point influence
            kill_distance: Distance at which attraction points are removed
            segment_length: Length of each growth segment
            width: Canvas width
            height: Canvas height
            root_position: Starting position (default: bottom center)
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.num_attractions = num_attractions
        self.attraction_distance = attraction_distance
        self.kill_distance = kill_distance
        self.segment_length = segment_length
        self.width = width
        self.height = height
        self.root_position = root_position or (width / 2, height)

    def generate_pattern(
        self,
        guidance_field: Callable[[np.ndarray, str], np.ndarray] | None = None,
        guidance_channel: str = "density",
        guidance_strength: float = 1.0,
        max_iterations: int = 1000,
    ) -> BranchNetwork:
        """Generate branching pattern.

        Args:
            guidance_field: Optional field function(points, channel) -> values
                          If provided, modulates growth behavior
            guidance_channel: Which channel to sample from guidance field
            guidance_strength: Scaling factor for guidance influence (0-1 typical)

        Returns:
            List of polylines representing branches

        Raises:
            RuntimeError: If generation fails (e.g., no growth occurred)
        """
        network = self.generate_network(
            guidance_field=guidance_field,
            guidance_channel=guidance_channel,
            guidance_strength=guidance_strength,
        )
        return network

    def generate_network(
        self,
        guidance_field: Callable[[np.ndarray, str], np.ndarray] | None = None,
        guidance_channel: str = "density",
        guidance_strength: float = 1.0,
    ) -> BranchNetwork:
        """Generate branching pattern as semantic network.

        Args:
            guidance_field: Optional field function(points, channel) -> values
                          If provided, modulates growth behavior
            guidance_channel: Which channel to sample from guidance field
            guidance_strength: Scaling factor for guidance influence (0-1 typical)
                             Higher values = stronger influence from guidance field

        Returns:
            BranchNetwork with full semantic information

        Raises:
            RuntimeError: If generation fails

        Example:
            # Unguided growth
            tree = gen.generate_network()

            # Guided by Voronoi cells
            voronoi = VoronoiTessellation(...).generate()
            guided_tree = gen.generate_network(
                guidance_field=voronoi.sample_field,
                guidance_channel='boundary_distance',
                guidance_strength=0.5
            )
        """
        # Generate attraction points
        attractions = self._generate_attractions()

        # Track nodes as they grow
        node_list = []
        node_list.append({"pos": np.array(self.root_position), "parent": -1, "timestamp": 0})

        # Grow iteratively
        max_iterations = 1000
        for iteration in range(1, max_iterations + 1):
            if len(attractions) == 0:
                break

            # Get current node positions
            node_positions = np.array([n["pos"] for n in node_list])

            # Find influences using vectorized function
            influenced_nodes, attr_indices = _compute_influences_vectorized(
                node_positions, attractions, self.attraction_distance
            )

            if len(influenced_nodes) == 0:
                break  # No more influences

            # Group attractions by influenced node
            influence_map = {}
            for node_idx, attr_idx in zip(influenced_nodes, attr_indices, strict=False):
                if node_idx not in influence_map:
                    influence_map[node_idx] = []
                influence_map[node_idx].append(attractions[attr_idx])

            # Grow new nodes from influenced nodes
            new_nodes = []
            for node_idx, attr_positions in influence_map.items():
                # Compute average direction to attractions
                parent_pos = node_positions[node_idx]
                directions = np.array(attr_positions) - parent_pos
                norms = np.linalg.norm(directions, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)  # Avoid division by zero
                directions = directions / norms

                avg_direction = np.mean(directions, axis=0)
                avg_direction = avg_direction / (np.linalg.norm(avg_direction) + 1e-10)

                # Apply guidance field if provided
                segment_len = self.segment_length
                if guidance_field is not None:
                    guidance_value = guidance_field(parent_pos.reshape(1, 2), guidance_channel)[0]
                    # Modulate segment length based on guidance field
                    # guidance_value typically in [0, 1], scale to affect growth
                    scale = 1.0 + guidance_strength * (guidance_value - 0.5)
                    segment_len = segment_len * np.clip(scale, 0.1, 2.0)

                # Create new node
                new_pos = parent_pos + avg_direction * segment_len

                # Keep within bounds
                new_pos[0] = np.clip(new_pos[0], 0, self.width)
                new_pos[1] = np.clip(new_pos[1], 0, self.height)

                new_nodes.append({"pos": new_pos, "parent": node_idx, "timestamp": iteration})

            if len(new_nodes) == 0:
                break

            node_list.extend(new_nodes)

            # Remove colonized attractions (those within kill distance of new nodes)
            new_node_positions = np.array([n["pos"] for n in new_nodes])
            tree = cKDTree(new_node_positions)

            to_remove = set()
            for i, attr_pos in enumerate(attractions):
                dist, _ = tree.query(attr_pos)
                if dist < self.kill_distance:
                    to_remove.add(i)

            attractions = np.array([a for i, a in enumerate(attractions) if i not in to_remove])

        # Validate generation
        if len(node_list) <= 1:
            raise RuntimeError(
                f"Space colonization failed: only {len(node_list)} node(s) generated. "
                "Try adjusting parameters (attraction_distance, num_attractions, etc.)"
            )

        # Convert to BranchNetwork
        return BranchNetwork.from_node_list(node_list, compute_branches=True)

    def _generate_attractions(self) -> np.ndarray:
        """Generate random attraction points in the canvas.

        Returns:
            (M, 2) array of attraction point positions
        """
        # Generate points in upper 2/3 of canvas (leave space for root)
        x = self.rng.uniform(0, self.width, self.num_attractions)
        y = self.rng.uniform(0, self.height * 0.66, self.num_attractions)

        return np.column_stack([x, y])
