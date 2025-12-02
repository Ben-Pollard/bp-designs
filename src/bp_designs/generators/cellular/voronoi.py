"""Voronoi tessellation pattern generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import Voronoi

from bp_designs.core.generator import Generator
from bp_designs.patterns.cells import Cells

if TYPE_CHECKING:
    pass


class VoronoiTessellation(Generator):
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

    def generate_pattern(self, guidance_field=None, **kwargs) -> Cells:
        """Generate Cells pattern.

        Args:
            guidance_field: Optional field function(points, channel) -> values (ignored for Voronoi)
            **kwargs: Additional parameters (ignored)

        Returns:
            Cells instance implementing Pattern interface

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

        # Return Cells instance
        bounds = (0, 0, self.width, self.height)
        return Cells(sites=sites, vor=vor, pattern_bounds=bounds, render_mode=self.render_mode)

    # Backward compatibility alias
    def generate(self) -> Cells:
        """Legacy method - use generate_pattern instead."""
        return self.generate_pattern()

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
