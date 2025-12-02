"""Geometry types and utilities."""

import numpy as np

from bp_designs.geometry.network import BranchNetwork

# Simple geometry type: list of polylines (Nx2 numpy arrays)
Geometry = list[np.ndarray]

__all__ = ["Geometry", "BranchNetwork"]
