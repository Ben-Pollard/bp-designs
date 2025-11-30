"""Voronoi tessellation pattern generator."""

import numpy as np
from scipy.spatial import Voronoi

from bp_designs.geometry import Geometry


class VoronoiTessellation:
    """Generate Voronoi (cellular) tessellation patterns.

    Creates natural-looking cellular patterns using Voronoi diagrams.
    Supports Lloyd's relaxation for more uniform cell distribution.
    """

    def __init__(
        self,
        seed: int = 0,
        num_sites: int = 50,
        relaxation_iterations: int = 0,
        render_mode: str = "edges",
        width: float = 100.0,
        height: float = 100.0,
        boundary_margin: float = 20.0,
    ):
        """Initialize Voronoi tessellation generator.

        Args:
            seed: Random seed for determinism
            num_sites: Number of Voronoi sites (cell centers)
            relaxation_iterations: Number of Lloyd's relaxation iterations (0 = none)
            render_mode: How to render ("edges", "cells", "both")
            width: Canvas width
            height: Canvas height
            boundary_margin: Margin outside canvas for mirror sites (prevents edge artifacts)
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.num_sites = num_sites
        self.relaxation_iterations = relaxation_iterations
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.boundary_margin = boundary_margin

        # Validate parameters
        if render_mode not in ["edges", "cells", "both"]:
            raise ValueError(f"render_mode must be 'edges', 'cells', or 'both', got {render_mode}")

    def generate(self) -> Geometry:
        """Generate Voronoi tessellation pattern.

        Returns:
            List of polylines representing Voronoi edges or cell boundaries

        Raises:
            RuntimeError: If generation fails
        """
        # Generate initial random sites
        sites = self._generate_sites()

        # Apply Lloyd's relaxation if requested
        if self.relaxation_iterations > 0:
            sites = self._apply_relaxation(sites, self.relaxation_iterations)

        # Add mirror sites around boundary to prevent edge artifacts
        sites_with_boundary = self._add_boundary_sites(sites)

        # Compute Voronoi diagram
        vor = Voronoi(sites_with_boundary)

        # Extract geometry based on render mode
        if self.render_mode == "edges":
            return self._extract_edges(vor)
        elif self.render_mode == "cells":
            return self._extract_cells(vor, sites)
        else:  # "both"
            edges = self._extract_edges(vor)
            cells = self._extract_cells(vor, sites)
            return edges + cells

    def _generate_sites(self) -> np.ndarray:
        """Generate random Voronoi sites within canvas.

        Returns:
            (N, 2) array of site positions
        """
        x = self.rng.uniform(0, self.width, self.num_sites)
        y = self.rng.uniform(0, self.height, self.num_sites)
        return np.column_stack([x, y])

    def _add_boundary_sites(self, sites: np.ndarray) -> np.ndarray:
        """Add mirror sites around boundary to prevent edge artifacts.

        Creates a virtual boundary by mirroring sites beyond canvas edges.

        Args:
            sites: (N, 2) array of original site positions

        Returns:
            (M, 2) array with original + boundary sites
        """
        margin = self.boundary_margin
        boundary_sites = []

        # Add sites around all four edges
        for site in sites:
            x, y = site

            # Mirror horizontally
            if x < self.width / 2:
                boundary_sites.append([x - margin, y])
            else:
                boundary_sites.append([x + margin, y])

            # Mirror vertically
            if y < self.height / 2:
                boundary_sites.append([x, y - margin])
            else:
                boundary_sites.append([x, y + margin])

        # Add corner sites
        boundary_sites.extend(
            [
                [-margin, -margin],
                [self.width + margin, -margin],
                [-margin, self.height + margin],
                [self.width + margin, self.height + margin],
            ]
        )

        return np.vstack([sites, np.array(boundary_sites)])

    def _apply_relaxation(self, sites: np.ndarray, iterations: int) -> np.ndarray:
        """Apply Lloyd's relaxation to create more uniform distribution.

        Iteratively moves each site to the centroid of its Voronoi cell.

        Args:
            sites: (N, 2) initial site positions
            iterations: Number of relaxation iterations

        Returns:
            (N, 2) relaxed site positions
        """
        current_sites = sites.copy()

        for _ in range(iterations):
            # Add boundary sites for proper computation
            sites_with_boundary = self._add_boundary_sites(current_sites)
            vor = Voronoi(sites_with_boundary)

            # Move each original site to its cell centroid
            new_sites = []
            for i in range(len(current_sites)):
                region_index = vor.point_region[i]
                region = vor.regions[region_index]

                # Skip infinite regions
                if -1 in region or len(region) == 0:
                    new_sites.append(current_sites[i])
                    continue

                # Compute centroid of polygon
                vertices = vor.vertices[region]
                centroid = np.mean(vertices, axis=0)

                # Keep within bounds
                centroid[0] = np.clip(centroid[0], 0, self.width)
                centroid[1] = np.clip(centroid[1], 0, self.height)

                new_sites.append(centroid)

            current_sites = np.array(new_sites)

        return current_sites

    def _extract_edges(self, vor: Voronoi) -> Geometry:
        """Extract Voronoi edges as polylines.

        Args:
            vor: Computed Voronoi diagram

        Returns:
            List of edge polylines (each edge as 2-point line)
        """
        edges = []

        for simplex in vor.ridge_vertices:
            # Skip infinite edges
            if -1 in simplex:
                continue

            # Get edge endpoints
            p1 = vor.vertices[simplex[0]]
            p2 = vor.vertices[simplex[1]]

            # Clip to canvas bounds
            if self._is_edge_visible(p1, p2):
                clipped = self._clip_line_to_bounds(p1, p2)
                if clipped is not None:
                    edges.append(np.array(clipped))

        return edges

    def _extract_cells(self, vor: Voronoi, original_sites: np.ndarray) -> Geometry:
        """Extract Voronoi cell boundaries as closed polylines.

        Only extracts cells for original sites (not boundary sites).

        Args:
            vor: Computed Voronoi diagram
            original_sites: Original site positions (before boundary sites added)

        Returns:
            List of cell boundary polylines
        """
        cells = []
        num_original = len(original_sites)

        for i in range(num_original):
            region_index = vor.point_region[i]
            region = vor.regions[region_index]

            # Skip infinite or empty regions
            if -1 in region or len(region) == 0:
                continue

            # Get vertices in order
            vertices = vor.vertices[region]

            # Clip to canvas bounds
            clipped = self._clip_polygon_to_bounds(vertices)
            if clipped is not None and len(clipped) >= 3:
                # Close the polygon
                closed = np.vstack([clipped, clipped[0]])
                cells.append(closed)

        return cells

    def _is_edge_visible(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check if edge intersects with canvas bounds.

        Args:
            p1, p2: Edge endpoints

        Returns:
            True if edge is at least partially visible
        """
        # Simple bounding box check
        margin = self.boundary_margin
        min_x, min_y = -margin, -margin
        max_x, max_y = self.width + margin, self.height + margin

        # Check if both points are completely outside same boundary
        if p1[0] < min_x and p2[0] < min_x:
            return False
        if p1[0] > max_x and p2[0] > max_x:
            return False
        if p1[1] < min_y and p2[1] < min_y:
            return False
        if p1[1] > max_y and p2[1] > max_y:
            return False

        return True

    def _clip_line_to_bounds(
        self, p1: np.ndarray, p2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Clip line segment to canvas bounds using Liang-Barsky algorithm.

        Args:
            p1, p2: Line endpoints

        Returns:
            Clipped endpoints or None if line is completely outside
        """
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1

        # Liang-Barsky algorithm
        t0, t1 = 0.0, 1.0

        # Check all four boundaries
        for edge in range(4):
            if edge == 0:  # Left
                p, q = -dx, x1 - 0
            elif edge == 1:  # Right
                p, q = dx, self.width - x1
            elif edge == 2:  # Bottom
                p, q = -dy, y1 - 0
            else:  # Top
                p, q = dy, self.height - y1

            if p == 0:  # Line parallel to boundary
                if q < 0:  # Line outside boundary
                    return None
            else:
                t = q / p
                if p < 0:  # Entering
                    t0 = max(t0, t)
                else:  # Leaving
                    t1 = min(t1, t)

        if t0 > t1:  # Line completely outside
            return None

        # Compute clipped points
        clipped_p1 = np.array([x1 + t0 * dx, y1 + t0 * dy])
        clipped_p2 = np.array([x1 + t1 * dx, y1 + t1 * dy])

        return (clipped_p1, clipped_p2)

    def _clip_polygon_to_bounds(self, vertices: np.ndarray) -> np.ndarray | None:
        """Clip polygon to canvas bounds using Sutherland-Hodgman algorithm.

        Args:
            vertices: (N, 2) polygon vertices

        Returns:
            Clipped vertices or None if polygon is completely outside
        """
        if len(vertices) == 0:
            return None

        # Clip against each edge sequentially
        output = vertices.copy()

        # Clip against each boundary: left (x=0), right (x=width), bottom (y=0), top (y=height)
        boundaries = [
            (0, 0, 1, 0),  # Left: normal = (1, 0), point = (0, 0)
            (self.width, 0, -1, 0),  # Right: normal = (-1, 0), point = (width, 0)
            (0, 0, 0, 1),  # Bottom: normal = (0, 1), point = (0, 0)
            (0, self.height, 0, -1),  # Top: normal = (0, -1), point = (0, height)
        ]

        for bx, by, nx, ny in boundaries:
            if len(output) == 0:
                return None

            input_list = output
            output = []

            for i in range(len(input_list)):
                current = input_list[i]
                next_vertex = input_list[(i + 1) % len(input_list)]

                # Check if vertices are inside (on positive side of boundary)
                current_inside = (current[0] - bx) * nx + (current[1] - by) * ny >= 0
                next_inside = (next_vertex[0] - bx) * nx + (next_vertex[1] - by) * ny >= 0

                if next_inside:
                    if not current_inside:
                        # Entering: add intersection point
                        intersection = self._line_plane_intersection(
                            current, next_vertex, (bx, by), (nx, ny)
                        )
                        if intersection is not None:
                            output.append(intersection)
                    # Add next vertex
                    output.append(next_vertex)
                elif current_inside:
                    # Leaving: add intersection point
                    intersection = self._line_plane_intersection(
                        current, next_vertex, (bx, by), (nx, ny)
                    )
                    if intersection is not None:
                        output.append(intersection)

            output = np.array(output) if output else np.array([])

        return output if len(output) >= 3 else None

    def _line_plane_intersection(
        self, p1: np.ndarray, p2: np.ndarray, plane_point: tuple, plane_normal: tuple
    ) -> np.ndarray | None:
        """Compute intersection of line segment with plane.

        Args:
            p1, p2: Line segment endpoints
            plane_point: Point on plane (bx, by)
            plane_normal: Plane normal vector (nx, ny)

        Returns:
            Intersection point or None if parallel
        """
        bx, by = plane_point
        nx, ny = plane_normal

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        denom = nx * dx + ny * dy
        if abs(denom) < 1e-10:  # Parallel
            return None

        t = (nx * (bx - p1[0]) + ny * (by - p1[1])) / denom

        # Clamp t to [0, 1] to stay within line segment
        t = max(0.0, min(1.0, t))

        return np.array([p1[0] + t * dx, p1[1] + t * dy])
