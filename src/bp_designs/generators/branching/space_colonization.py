"""Space Colonization algorithm - vectorized implementation with semantic preservation."""

import numpy as np
from einops import rearrange, reduce
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon

from bp_designs.core.directions import DirectionVectors, PairwiseCoordinateVectors
from bp_designs.core.generator import Generator
from bp_designs.core.geometry import Canvas, Point, Polygon
from bp_designs.patterns.network import BranchNetwork


class SpaceColonization(Generator):
    """Generate branching patterns using Space Colonization algorithm.

    Vectorized implementation preserving semantic information (hierarchy, branch IDs).

    This algorithm simulates growth toward "attraction points" to create
    natural-looking vein or root-like structures.
    """

    def __init__(
        self,
        canvas: Canvas,
        root_position: Point,
        initial_boundary: Polygon,
        final_boundary: Polygon,
        seed: int = 0,
        num_attractions: int = 500,
        kill_distance: float = 5.0,
        segment_length: float = 2.0,
        max_iterations: int = 1000,
    ):
        """Initialize Space Colonization generator.

        Args:
            canvas: defines the coordinate system
            root_position: Starting position (default: bottom center)
            initial_boundary: Polygon for initial attraction points
            final_boundary: Polygon for maximum growth region
            seed: Random seed for determinism
            num_attractions: Number of attraction points (growth targets)
            kill_distance: Distance at which attraction points are removed
            segment_length: Length of each growth segment
            max_iterations: Maximum number of growth iterations
        """
        self.rng = np.random.default_rng(seed)

        self.num_attractions = num_attractions
        self.kill_distance = kill_distance
        self.segment_length = segment_length
        self.root_position = root_position
        self.canvas = canvas
        # Convert Point to array for positions
        self.root_position_array = np.array([root_position.x, root_position.y], dtype=float)
        # Store boundaries and iteration limit
        self.initial_boundary = initial_boundary
        self.final_boundary = final_boundary
        self.max_iterations = max_iterations

    def generate_pattern(self, **kwargs) -> BranchNetwork:
        """Generate branching pattern using stored parameters.

        Args:
            **kwargs: For compatibility with Generator interface (ignored)

        Returns:
            BranchNetwork representing the generated branching pattern

        Raises:
            RuntimeError: If generation fails (e.g., no growth occurred)
        """
        initial_timestamp = 0
        network_previous = self._initialize_network(initial_timestamp)

        attractions = self._initialize_attractions(0, self.initial_boundary)

        for _ in range(self.max_iterations):
            network, attractions = self._iterate(network_previous, attractions, self.final_boundary)
            if len(attractions) == 0:
                break
            if network.num_nodes == network_previous.num_nodes:
                break
            network_previous = network

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

        while len(points) < num_attractions:
            # Generate candidate points in bounding box
            x = self.rng.uniform(xmin, xmax, batch_size)
            y = self.rng.uniform(ymin, ymax, batch_size)
            candidates = np.column_stack([x, y])  # (batch_size, 2)

            # Check which points are inside polygon using shapely
            # Simple loop is fine for now
            for point in candidates:
                if shapely_poly.contains(ShapelyPoint(point)):
                    points.append(point)
                    if len(points) >= num_attractions:
                        break

        return np.array(points[:num_attractions])

    def _attraction_vectors(
        self, network: BranchNetwork, attractions: np.ndarray
    ) -> PairwiseCoordinateVectors:
        """Compute vectors from network nodes to attraction points.

        Args:
            network: BranchNetwork with N nodes
            attractions: (M, 2) array of attraction point positions

        Returns:
            AttractionVectors containing pairwise vectors, distances, and directions
        """
        attr_vectors = rearrange(attractions, "m c -> 1 m c") - rearrange(
            network.positions, "n c -> n 1 c"
        )  # (n,m,c)
        attr_norms = np.linalg.norm(attr_vectors, axis=2)  # (n,m)
        attr_directions = attr_vectors / rearrange(attr_norms, "n m -> n m 1")  # (n,m)
        return PairwiseCoordinateVectors(
            vectors=attr_vectors, norms=attr_norms, directions=attr_directions
        )

    def _growth_vectors(self, attraction_vectors: PairwiseCoordinateVectors) -> DirectionVectors:
        """Compute growth vectors from attraction vectors.

        Args:
            attraction_vectors: PairwiseCoordinateVectors representing attractions

        Returns:
            (2, N) array of growth vectors for each node
        """
        if attraction_vectors.empty:
            return DirectionVectors(
                vectors=np.empty([0, 2]), norms=np.empty([0]), directions=np.empty([0, 2])
            )
        # Nodes grow if they are nearest to a source
        # Nodes are influenced only by the sources they are closest to
        closest_nodes = attraction_vectors.norms.argmin(axis=0)  # (M,)
        num_nodes = attraction_vectors.norms.shape[0]
        closest_mask = closest_nodes == np.arange(num_nodes)[:, None]
        closest_directions = np.where(
            closest_mask[..., None], attraction_vectors.directions, np.nan
        )

        node_directions = reduce(closest_directions, "n m coord -> n coord", np.nanmean)
        norms = rearrange(np.linalg.norm(node_directions, axis=1), "n -> n 1")
        normalised_node_directions = node_directions / norms

        growth_vectors = normalised_node_directions * self.segment_length
        growth_norms = np.linalg.norm(growth_vectors, axis=1)
        return DirectionVectors(
            vectors=growth_vectors, norms=growth_norms, directions=normalised_node_directions
        )

    def _iterate(
        self,
        network: BranchNetwork,
        attractions: np.ndarray,
        final_boundary: Polygon,
    ) -> tuple[BranchNetwork, np.ndarray]:
        """ """
        # For now, use final_boundary as current boundary
        # TODO: Compute proper current boundary that encapsulates nodes + segment length
        current_boundary = final_boundary
        # Place new attractions
        new_attractions = self._initialize_attractions(self.num_attractions, current_boundary)

        # Check neighbourhoods for inclusion of previously placed attractions
        if attractions.size > 0:
            new_attr_vectors = rearrange(attractions, "m c -> 1 m c") - rearrange(
                new_attractions, "mn c -> mn 1 c"
            )  # (mn m c)
            new_attr_norms = np.linalg.norm(new_attr_vectors, axis=2)  # (mn, m)
            new_attr_selection = reduce(new_attr_norms, "mn m -> mn", "min") > self.kill_distance
            attractions = np.vstack([attractions, new_attractions[new_attr_selection, :]])
        else:
            attractions = new_attractions

        # Calculate vectors between nodes and sources
        attraction_vectors = self._attraction_vectors(network, attractions)

        # Remove colonized attractions (those within kill distance of nodes)
        kill_mask = reduce(attraction_vectors.norms, "n m -> m", "min") > self.kill_distance
        attractions = attractions[kill_mask, :]

        # Calculate the attraction vectors
        attraction_vectors = self._attraction_vectors(network, attractions)

        # Filter tree down to just the growing nodes
        growth_node_indices = np.unique(attraction_vectors.norms.argmin(axis=0))
        growing_nodes = network.get_nodes(growth_node_indices)
        attraction_vectors = attraction_vectors.get_n(growth_node_indices)

        # Calculate growth vectors of growing nodes
        growth_vectors = self._growth_vectors(attraction_vectors)

        # Create updated network with new nodes
        new_positions = growing_nodes.positions + growth_vectors.vectors
        num_new_nodes = new_positions.shape[0]
        next_node_id = network.node_ids.max() + 1
        new_node_ids = np.arange(next_node_id, next_node_id + num_new_nodes)
        new_timestamps = np.full(num_new_nodes, network.timestamps.max() + 1, dtype=np.int16)

        updated_network = BranchNetwork(
            node_ids=np.hstack([network.node_ids, new_node_ids]),
            positions=np.vstack([network.positions, new_positions]),
            parents=np.hstack([network.parents, growing_nodes.node_ids]),
            timestamps=np.hstack([network.timestamps, new_timestamps]),
        )

        return updated_network, attractions
