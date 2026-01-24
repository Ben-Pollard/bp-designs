"""Space Colonization algorithm - vectorized implementation with semantic preservation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from bp_designs.core.directions import DirectionVectors
from bp_designs.core.generator import Generator
from bp_designs.core.geometry import Canvas, Point, Polygon
from bp_designs.core.pattern import Pattern
from bp_designs.patterns.network import BranchNetwork
from bp_designs.patterns.network.refinement import NetworkRefinementStrategy

if TYPE_CHECKING:
    from bp_designs.patterns.network.refinement import NetworkRefinementStrategy
    from bp_designs.patterns.organs import OrganPattern


class SpaceColonization(Generator):
    """Generate branching patterns using Space Colonization algorithm.

    Vectorized implementation preserving semantic information (hierarchy, branch IDs).

    This algorithm simulates growth toward "attraction points" to create
    natural-looking vein or root-like structures.
    """

    def __init__(
        self,
        canvas: Canvas,
        root_position: Point | Pattern,
        initial_boundary: Pattern,
        final_boundary: Pattern,
        seed: int = 0,
        num_attractions: int = 500,
        kill_distance: float = 5.0,
        segment_length: float = 2.0,
        boundary_expansion: float = 10.0,
        max_iterations: int = 1000,
        refinement_strategy: NetworkRefinementStrategy = None,
        organ_template: OrganPattern | None = None,
        organ_distribution: str | Any = "terminal",
    ):
        """Initialize Space Colonization generator.

        Args:
            canvas: defines the coordinate system
            root_position: Starting position (default: bottom center)
            initial_boundary: Pattern defining region for initial attraction points
            final_boundary: Pattern defining maximum growth region
            seed: Random seed for determinism
            num_attractions: Number of attraction points (growth targets)
            kill_distance: Distance at which attraction points are removed
            segment_length: Length of each growth segment
            boundary_expansion: Buffer distance for organic boundary growth
            max_iterations: Maximum number of growth iterations
        """
        self.rng = np.random.default_rng(seed)

        self.num_attractions = num_attractions
        self.kill_distance = kill_distance
        self.segment_length = segment_length
        self.boundary_expansion = boundary_expansion
        self.root_position_pattern = root_position
        self.canvas = canvas
        self.refinement_strategy = refinement_strategy
        # Store boundary patterns
        self.initial_boundary_pattern = initial_boundary
        self.final_boundary_pattern = final_boundary

        # Resolve patterns to geometry using canvas
        if isinstance(root_position, Pattern):
            resolved_root = root_position.to_geometry(canvas)
            if not isinstance(resolved_root, Point):
                raise ValueError(f"root_position pattern must resolve to Point, got {type(resolved_root)}")
            self.root_position_array = np.array([resolved_root.x, resolved_root.y], dtype=float)
        else:
            self.root_position_array = np.array([root_position.x, root_position.y], dtype=float)

        # Resolve boundaries to polygons
        initial_geom = initial_boundary.to_geometry(canvas)
        if hasattr(initial_geom, "polylines"):  # Polyline
            self.initial_boundary = Polygon(coords=initial_geom.polylines[0])
        else:
            raise ValueError(f"initial_boundary must resolve to Polyline/Polygon, got {type(initial_geom)}")

        final_geom = final_boundary.to_geometry(canvas)
        if hasattr(final_geom, "polylines"):  # Polyline
            self.final_boundary = Polygon(coords=final_geom.polylines[0])
        else:
            raise ValueError(f"final_boundary must resolve to Polyline/Polygon, got {type(final_geom)}")
        self.max_iterations = max_iterations
        self.organ_template = organ_template
        self.organ_distribution = organ_distribution

    def generate_pattern(
        self, **render_params
    ) -> BranchNetwork:
        """Generate branching pattern using stored parameters.

        Args:
            refinement_strategy: Optional strategy to refine the network before organ attachment.
            **render_params: Rendering parameters to store in the resulting Pattern.

        Returns:
            BranchNetwork representing the generated branching pattern

        Raises:
            RuntimeError: If generation fails (e.g., no growth occurred)
        """
        initial_timestamp = 0
        network_previous = self._initialize_network(initial_timestamp)

        attractions = self._initialize_attractions(self.num_attractions, self.initial_boundary)

        for _ in range(self.max_iterations):
            network, attractions = self._iterate(network_previous, attractions)
            if attractions.size == 0:
                break
            if network.num_nodes == network_previous.num_nodes:
                break
            network_previous = network

        # Ensure canvas and pattern_bounds are set on the final network
        network.canvas = self.canvas
        network.pattern_bounds = self.canvas.bounds()

        # Apply refinement if strategy provided
        if self.refinement_strategy is not None:
            network = self.refinement_strategy.apply(network)

        # Attach organs to the final network if template provided
        if self.organ_template is not None:
            network.organ_template = self.organ_template
            network.organ_distribution = self.organ_distribution
            # Pass distribution_params if they exist in render_params
            dist_params = render_params.get("distribution_params", {})
            network.attach_organs(self.organ_template, self.organ_distribution, distribution_params=dist_params)

        network.render_params = render_params
        return network

    def _initialize_network(self, timestamp: int) -> BranchNetwork:
        """Initialize network with root node.

        Args:
            timestamp: Initial timestamp for root node (typically 0)

        Returns:
            BranchNetwork with single root node
        """
        network = BranchNetwork(
            node_ids=np.array([0], dtype=np.int16),  # (N,) - start node IDs at 0
            positions=self.root_position_array.reshape(1, 2),  # (N,2) - root position as array
            parents=np.array([-1], dtype=np.int16),  # (N,) - root has no parent
            timestamps=np.array([timestamp], dtype=np.int16),  # (N,) - root timestamp
            canvas=self.canvas,
            pattern_bounds=self.canvas.bounds(),
        )
        return network

    def _initialize_attractions(self, num_attractions: int, boundary: Polygon) -> np.ndarray:
        """Generate attraction points inside boundary polygon.

        Uses rejection sampling: generate points in bounding box,
        keep only those inside polygon using shapely.

        Args:
            num_attractions: Number of attraction points to generate
            boundary: Polygon defining region

        Returns:
            (N, 2) array of attraction point positions inside boundary
        """
        if num_attractions <= 0:
            return np.array([], dtype=float).reshape(0, 2)

        # Get bounding box for efficient sampling
        bounds = boundary.bounds()
        xmin, ymin, xmax, ymax = bounds

        # Create shapely polygon for containment checks
        shapely_poly = ShapelyPolygon(boundary.coords)

        points = []
        batch_size = min(num_attractions * 2, 1000)

        # Safety break to avoid infinite loop if boundary is too small
        max_attempts = 100
        attempts = 0

        while len(points) < num_attractions and attempts < max_attempts:
            attempts += 1
            # Generate candidate points in bounding box
            x = self.rng.uniform(xmin, xmax, batch_size)
            y = self.rng.uniform(ymin, ymax, batch_size)
            candidates = np.column_stack([x, y])  # (batch_size, 2)

            # Vectorized containment check using shapely.contains
            import shapely

            candidate_points = shapely.points(candidates)
            mask = shapely.contains(shapely_poly, candidate_points)
            valid_points = candidates[mask]
            points.extend(valid_points)

        return np.array(points[:num_attractions])

    def _compute_current_boundary(self, network: BranchNetwork) -> Polygon:
        """Compute organic boundary based on current network extent.

        Creates convex hull around nodes and buffers by expansion distance,
        clipped to final boundary.

        Args:
            network: Current branch network

        Returns:
            Polygon defining current growth boundary
        """
        # Compute convex hull of all nodes
        import shapely

        points = shapely.points(network.positions)
        hull = shapely.convex_hull(shapely.multipoints(points))

        # Buffer by expansion distance
        expanded = hull.buffer(self.boundary_expansion)

        # Clip to final boundary
        final_shapely = ShapelyPolygon(self.final_boundary.coords)
        clipped = expanded.intersection(final_shapely)

        # Convert back to our Polygon type
        if clipped.is_empty:
            # Fallback to final boundary if intersection fails
            return self.final_boundary

        # Handle MultiPolygon by taking convex hull
        if clipped.geom_type == "MultiPolygon":
            clipped = clipped.convex_hull
            # After convex hull, may still be Point/LineString if degenerate
            if clipped.is_empty:
                return self.final_boundary

        # Extract coordinates from shapely polygon (handles Polygon only)
        try:
            coords = np.array(clipped.exterior.coords)
            return Polygon(coords=coords)
        except AttributeError:
            # clipped is not a Polygon (e.g., Point, LineString, GeometryCollection)
            # Fallback to final boundary
            return self.final_boundary

    def _attraction_vectors(
        self, network: BranchNetwork, attractions: np.ndarray, influence_distance: float = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute vectors from network nodes to attraction points.

        Args:
            network: BranchNetwork with N nodes
            attractions: (M, 2) array of attraction point positions
            influence_distance: Optional maximum distance for attraction

        Returns:
            tuple of (closest_node_indices, dists, vectors)
        """
        if attractions.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float).reshape(0, 2)

        # Optimization: Use KDTree to find closest nodes for each attraction point.
        # This avoids the O(N*M) distance matrix.
        try:
            from scipy.spatial import cKDTree

            node_tree = cKDTree(network.positions)

            # For each attraction point, find the closest node
            # dists: (M,), closest_node_indices: (M,)
            dists, closest_node_indices = node_tree.query(attractions, k=1)

            # Compute vectors only for the closest pairs
            vectors = attractions - network.positions[closest_node_indices]

            return closest_node_indices, dists, vectors

        except ImportError:
            # Fallback to dense broadcasting if scipy is missing
            attr_vectors = attractions[None, :, :] - network.positions[:, None, :]
            attr_norms = np.sqrt(np.sum(attr_vectors**2, axis=2))
            closest_node_indices = np.argmin(attr_norms, axis=0)
            dists = np.min(attr_norms, axis=0)
            vectors = attractions - network.positions[closest_node_indices]
            return closest_node_indices, dists, vectors

    def _growth_vectors(
        self, closest_node_indices: np.ndarray, dists: np.ndarray, vectors: np.ndarray, num_nodes: int
    ) -> DirectionVectors:
        """Compute growth vectors from attraction data.

        Args:
            closest_node_indices: (M,) index of closest node for each attraction point
            dists: (M,) distance to closest node
            vectors: (M, 2) vector from node to attraction point
            num_nodes: Total number of nodes in network

        Returns:
            DirectionVectors for each node
        """
        if dists.size == 0:
            return DirectionVectors(
                vectors=np.empty([0, 2]), norms=np.empty([0]), directions=np.empty([0, 2])
            )

        # Initialize node directions
        node_directions = np.zeros((num_nodes, 2))

        # Compute directions for attraction points
        with np.errstate(divide="ignore", invalid="ignore"):
            mask = dists > 0
            directions = np.zeros_like(vectors)
            directions[mask] = vectors[mask] / dists[mask, None]

        # For each attraction point, add its direction to its closest node
        np.add.at(node_directions, closest_node_indices, directions)

        # Normalize node directions
        norms = np.sqrt(np.sum(node_directions**2, axis=1))[:, None]

        with np.errstate(divide="ignore", invalid="ignore"):
            normalised_node_directions = np.where(norms > 0, node_directions / norms, 0.0)

        growth_vectors = normalised_node_directions * self.segment_length
        growth_norms = np.sqrt(np.sum(growth_vectors**2, axis=1))

        return DirectionVectors(
            vectors=growth_vectors, norms=growth_norms, directions=normalised_node_directions
        )

    def _iterate(
        self,
        network: BranchNetwork,
        attractions: np.ndarray,
    ) -> tuple[BranchNetwork, np.ndarray]:
        """Perform one iteration of growth."""
        # Compute current boundary based on network extent
        current_boundary = self._compute_current_boundary(network)

        # Place new attractions within current boundary
        new_attractions = self._initialize_attractions(self.num_attractions, current_boundary)

        # Check neighbourhoods for inclusion of previously placed attractions
        if attractions.size > 0:
            try:
                from scipy.spatial import cKDTree

                tree = cKDTree(attractions)
                dist, _ = tree.query(new_attractions, k=1)
                new_attr_selection = dist > self.kill_distance
            except ImportError:
                new_attr_vectors = new_attractions[None, :, :] - attractions[:, None, :]
                new_attr_norms = np.sqrt(np.sum(new_attr_vectors**2, axis=2))
                new_attr_selection = np.min(new_attr_norms, axis=0) > self.kill_distance

            attractions = np.vstack([attractions, new_attractions[new_attr_selection, :]])
        else:
            attractions = new_attractions

        # Remove colonized attractions (those within kill distance of nodes)
        try:
            from scipy.spatial import cKDTree

            node_tree = cKDTree(network.positions)
            dist, _ = node_tree.query(attractions, k=1)
            kill_mask = dist > self.kill_distance
        except ImportError:
            # Fallback to dense if scipy missing
            attr_vectors = attractions[None, :, :] - network.positions[:, None, :]
            attr_norms = np.sqrt(np.sum(attr_vectors**2, axis=2))
            kill_mask = np.min(attr_norms, axis=0) > self.kill_distance

        attractions = attractions[kill_mask, :]

        # Calculate the attraction data using KDTree
        closest_node_indices, dists, vectors = self._attraction_vectors(network, attractions)

        if dists.size == 0:
            return network, attractions

        # Filter down to just the growing nodes (those that are closest to at least one attraction point)
        growth_node_indices = np.unique(closest_node_indices)
        growing_nodes = network.get_nodes(growth_node_indices)

        # Calculate growth vectors
        growth_vectors = self._growth_vectors(closest_node_indices, dists, vectors, network.num_nodes)

        # We only care about growth vectors for the growing nodes
        active_growth_vectors = growth_vectors.vectors[growth_node_indices]

        # Create updated network with new nodes
        new_positions = growing_nodes.positions + active_growth_vectors

        num_new_nodes = new_positions.shape[0]
        next_node_id = network.node_ids.max() + 1
        new_node_ids = np.arange(next_node_id, next_node_id + num_new_nodes)
        new_timestamps = np.full(num_new_nodes, network.timestamps.max() + 1, dtype=np.int16)

        updated_network = BranchNetwork(
            node_ids=np.hstack([network.node_ids, new_node_ids]),
            positions=np.vstack([network.positions, new_positions]),
            parents=np.hstack([network.parents, growing_nodes.node_ids]),
            timestamps=np.hstack([network.timestamps, new_timestamps]),
            canvas=network.canvas,
            pattern_bounds=network.pattern_bounds,
        )

        return updated_network, attractions
