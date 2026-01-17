"""Cell data structure for semantic and vectorized operations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import KDTree, Voronoi

from bp_designs.core.geometry import Polyline
from bp_designs.core.pattern import Pattern
from bp_designs.core.renderer import RenderingContext, RenderStyle


class CellsStyle(RenderStyle):
    """Structured rendering parameters for Voronoi cells."""

    stroke_width: float = 0.5
    stroke_color: str = "#000000"
    fill: str | None = None
    stroke_linecap: str = "round"
    stroke_linejoin: str = "round"


@dataclass
class Cells(Pattern):
    """Voronoi tessellation pattern implementing Pattern interface.



    Field Channels:


        - 'cell_id': ID of cell containing each point


        - 'boundary_distance': Distance to nearest cell boundary


        - 'cell_size': Relative size of containing cell


        - 'center_distance': Distance to cell center
    """

    sites: np.ndarray  # (N, 2) - Voronoi site positions

    vor: Voronoi  # Voronoi diagram object from scipy.spatial

    pattern_bounds: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)

    render_mode: str = "edges"  # How to render: "edges", "cells", "both"

    # Cached spatial indices

    _site_kdtree: KDTree | None = field(default=None, init=False, repr=False)

    _edge_kdtree: KDTree | None = field(default=None, init=False, repr=False)

    def sample_field(self, points: np.ndarray, channel: str) -> np.ndarray:
        """Sample Voronoi pattern as a field.



        Args:


            points: (N, 2) array of query positions


            channel: One of: 'cell_id', 'boundary_distance', 'cell_size', 'center_distance'



        Returns:


            (N,) array of values



        Raises:


            ValueError: If channel is unknown
        """

        if channel == "cell_id":
            # Which cell contains each point

            return self._get_cell_ids(points)

        elif channel == "boundary_distance":
            # Distance to nearest cell boundary

            return self._get_boundary_distances(points)

        elif channel == "cell_size":
            # Relative size of containing cell

            return self._get_cell_sizes(points)

        elif channel == "center_distance":
            # Distance to cell center

            return self._get_center_distances(points)

        else:
            raise ValueError(
                f"Unknown channel '{channel}'. Available: {list(self.available_channels().keys())}"
            )

    def available_channels(self) -> dict[str, str]:
        """Return available field channels with descriptions."""

        return {
            "cell_id": "ID of cell containing each point",
            "boundary_distance": "Distance to nearest cell boundary",
            "cell_size": "Relative size of containing cell (0-1)",
            "center_distance": "Distance to cell center",
        }

    def to_geometry(self) -> Polyline:
        """Convert Voronoi pattern to geometry.


        Returns:

            Geometry based on render_mode: edges, cells, or both
        """

        if self.render_mode == "edges":
            polylines = self._extract_edges()

        elif self.render_mode == "cells":
            polylines = self._extract_cells()

        else:  # "both"
            edges = self._extract_edges()

            cells = self._extract_cells()

            polylines = edges + cells

        return Polyline(polylines=polylines)

    def render(
        self,
        context: RenderingContext,
        style: CellsStyle | None = None,
        **kwargs,
    ):
        """Render Voronoi cells into the provided context."""
        # Merge style and kwargs
        if style is None:
            style = CellsStyle.from_dict(kwargs)
        else:
            style_dict = style.model_dump()
            style_dict.update(kwargs)
            style = CellsStyle.from_dict(style_dict)

        # Get geometry polylines
        geometry = self.to_geometry()

        svg_attrs = style.get_svg_attributes()

        # Draw each polyline
        for polyline in geometry.polylines:
            if len(polyline) < 2:
                continue

            # Convert to list of (x, y) tuples
            points = [(float(x), float(y)) for x, y in polyline]

            if len(polyline) == 2:
                # Edge - draw as line
                line = context.dwg.line(
                    start=points[0],
                    end=points[1],
                    stroke=style.stroke_color,
                    stroke_width=style.stroke_width,
                    stroke_linecap=style.stroke_linecap,
                    fill="none",
                    **svg_attrs,
                )
                context.add(line)
            else:
                # Cell polygon - draw as polygon
                poly = context.dwg.polygon(
                    points=points,
                    stroke=style.stroke_color,
                    stroke_width=style.stroke_width,
                    stroke_linecap=style.stroke_linecap,
                    stroke_linejoin=style.stroke_linejoin,
                    fill=style.fill if style.fill is not None else "none",
                    **svg_attrs,
                )
                context.add(poly)

    def bounds(self) -> tuple[float, float, float, float]:
        """Return pattern bounds."""

        return self.pattern_bounds

    @property
    def site_kdtree(self) -> KDTree:
        """Lazy-initialized KDTree for site queries."""

        if self._site_kdtree is None:
            self._site_kdtree = KDTree(self.sites)

        return self._site_kdtree

    @property
    def edge_kdtree(self) -> KDTree:
        """Lazy-initialized KDTree for edge queries."""

        if self._edge_kdtree is None:
            # Extract all edge points

            edge_points = []

            for simplex in self.vor.ridge_vertices:
                if -1 not in simplex:
                    edge_points.extend(
                        [self.vor.vertices[simplex[0]], self.vor.vertices[simplex[1]]]
                    )

            if edge_points:
                self._edge_kdtree = KDTree(np.array(edge_points))
            else:
                # Fallback to sites if no edges

                self._edge_kdtree = KDTree(self.sites)

        return self._edge_kdtree

    def _get_cell_ids(self, points: np.ndarray) -> np.ndarray:
        """Get cell ID for each point."""

        # Find nearest site = cell ID

        _, indices = self.site_kdtree.query(points)

        return indices.astype(float)

    def _get_boundary_distances(self, points: np.ndarray) -> np.ndarray:
        """Get distance to nearest cell boundary."""

        distances, _ = self.edge_kdtree.query(points)
        return distances

    def _get_cell_sizes(self, points: np.ndarray) -> np.ndarray:
        """Get relative cell sizes."""

        # Find containing cell

        self._get_cell_ids(points)

        # Compute cell areas (simplified - use distance to nearest neighbors)

        _, nearest_dists = self.site_kdtree.query(points, k=2)

        # Use ratio of distances as proxy for cell size

        sizes = nearest_dists[:, 1] / (nearest_dists[:, 0] + 1e-10)

        return np.clip(sizes, 0, 1)

    def _get_center_distances(self, points: np.ndarray) -> np.ndarray:
        """Get distance to cell center."""

        # Find nearest site = cell center

        distances, _ = self.site_kdtree.query(points)
        return distances

    def _extract_edges(self) -> Polyline:
        """Extract Voronoi edges as polylines."""

        edges = []

        for simplex in self.vor.ridge_vertices:
            # Skip infinite edges

            if -1 in simplex:
                continue

            # Get edge endpoints

            p1 = self.vor.vertices[simplex[0]]

            p2 = self.vor.vertices[simplex[1]]

            # Clip to bounds

            xmin, ymin, xmax, ymax = self.pattern_bounds

            if self._is_edge_visible(p1, p2, xmin, ymin, xmax, ymax):
                clipped = self._clip_line_to_bounds(p1, p2, xmin, ymin, xmax, ymax)

                if clipped is not None:
                    edges.append(np.array(clipped))

        return edges

    def _extract_cells(self) -> Polyline:
        """Extract Voronoi cell boundaries as closed polylines.



        Only extracts cells for original sites (not boundary sites).



        Returns:


            List of cell boundary polylines
        """

        cells = []

        num_original = len(self.sites)

        for i in range(num_original):
            region_index = self.vor.point_region[i]

            region = self.vor.regions[region_index]

            # Skip infinite or empty regions

            if -1 in region or len(region) == 0:
                continue

            # Get vertices in order

            vertices = self.vor.vertices[region]

            # Clip to bounds

            xmin, ymin, xmax, ymax = self.pattern_bounds

            clipped = self._clip_polygon_to_bounds(vertices, xmin, ymin, xmax, ymax)

            if clipped is not None and len(clipped) >= 3:
                # Close the polygon

                closed = np.vstack([clipped, clipped[0]])

                cells.append(closed)

        return cells

    def _is_edge_visible(
        self, p1: np.ndarray, p2: np.ndarray, xmin: float, ymin: float, xmax: float, ymax: float
    ) -> bool:
        """Check if edge intersects with bounds."""

        # Simple bounding box check

        if p1[0] < xmin and p2[0] < xmin:
            return False

        if p1[0] > xmax and p2[0] > xmax:
            return False

        if p1[1] < ymin and p2[1] < ymin:
            return False

        if p1[1] > ymax and p2[1] > ymax:
            return False

        return True

    def _clip_line_to_bounds(
        self, p1: np.ndarray, p2: np.ndarray, xmin: float, ymin: float, xmax: float, ymax: float
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Clip line segment to bounds using Liang-Barsky algorithm."""

        x1, y1 = p1

        x2, y2 = p2

        dx = x2 - x1

        dy = y2 - y1

        # Liang-Barsky algorithm

        t0, t1 = 0.0, 1.0

        # Check all four boundaries

        for edge in range(4):
            if edge == 0:  # Left
                p, q = -dx, x1 - xmin

            elif edge == 1:  # Right
                p, q = dx, xmax - x1

            elif edge == 2:  # Bottom
                p, q = -dy, y1 - ymin

            else:  # Top
                p, q = dy, ymax - y1

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

    def _clip_polygon_to_bounds(
        self, vertices: np.ndarray, xmin: float, ymin: float, xmax: float, ymax: float
    ) -> np.ndarray | None:
        """Clip polygon to bounds using Sutherland-Hodgman algorithm.



        Args:


            vertices: (N, 2) polygon vertices


            xmin, ymin, xmax, ymax: Bounds to clip against



        Returns:


            Clipped vertices or None if polygon is completely outside
        """

        if len(vertices) == 0:
            return None

        # Clip against each edge sequentially

        output = vertices.copy()

        # Clip against each boundary: left, right, bottom, top

        boundaries = [
            (xmin, 0, 1, 0),  # Left: normal = (1, 0)
            (xmax, 0, -1, 0),  # Right: normal = (-1, 0)
            (0, ymin, 0, 1),  # Bottom: normal = (0, 1)
            (0, ymax, 0, -1),  # Top: normal = (0, -1)
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

    def __str__(self) -> str:
        """Return human-readable string representation."""
        num_sites = len(self.sites) if self.sites is not None else 0
        return f"VoronoiCells(N={num_sites}, mode={self.render_mode})"

    def __eq__(self, other: object) -> bool:
        """Equality based on sites and pattern bounds."""
        if not isinstance(other, Cells):
            return False
        # Compare sites arrays
        if self.sites.shape != other.sites.shape:
            return False
        if not np.array_equal(self.sites, other.sites):
            return False
        # Compare pattern bounds
        if self.pattern_bounds != other.pattern_bounds:
            return False
        # Compare render mode
        if self.render_mode != other.render_mode:
            return False
        return True

    def __hash__(self) -> int:
        """Hash based on sites data and pattern bounds."""
        # Convert sites to tuple of tuples for hashing
        if self.sites.size == 0:
            sites_hash = hash(())
        else:
            sites_tuple = tuple(tuple(map(float, point)) for point in self.sites)
            sites_hash = hash(sites_tuple)
        # Combine with pattern bounds and render mode
        return hash((sites_hash, self.pattern_bounds, self.render_mode))
