"""Parameter space definition and grid generation for experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np


@dataclass
class ParameterSpace:
    """Define parameter ranges for experimentation.

    Supports multiple sampling strategies:
    - Linear spacing (linspace)
    - Logarithmic spacing (logspace)
    - Explicit value lists
    """

    name: str
    ranges: dict[str, list[Any] | tuple[float, float, int]]
    fixed: dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize fixed parameters if not provided."""
        if self.fixed is None:
            self.fixed = {}

    def expand_range(self, param_name: str, spec: Any) -> list[Any]:
        """Expand a parameter specification into a list of values.

        Args:
            param_name: Parameter name (for error messages)
            spec: Either a list of explicit values or tuple (min, max, steps)

        Returns:
            List of parameter values to test
        """
        if isinstance(spec, list):
            # Explicit values
            return spec
        elif isinstance(spec, tuple) and len(spec) == 3:
            # (min, max, steps) - linear spacing
            return list(np.linspace(spec[0], spec[1], spec[2]))
        else:
            raise ValueError(
                f"Invalid parameter spec for '{param_name}': {spec}. "
                "Expected list [v1, v2, ...] or tuple (min, max, steps)"
            )

    def to_grid(self) -> ParameterGrid:
        """Generate all parameter combinations as a grid.

        Returns:
            ParameterGrid with all combinations
        """
        # Expand all ranges
        expanded = {}
        for param_name, spec in self.ranges.items():
            expanded[param_name] = self.expand_range(param_name, spec)

        # Generate cartesian product
        param_names = list(expanded.keys())
        param_values = [expanded[name] for name in param_names]

        combinations = []
        for values in product(*param_values):
            combo = dict(zip(param_names, values, strict=False))
            # Add fixed parameters
            combo.update(self.fixed)
            combinations.append(combo)

        return ParameterGrid(
            space_name=self.name,
            param_names=param_names,
            fixed_params=self.fixed,
            combinations=combinations,
        )


@dataclass
class ParameterGrid:
    """Grid of parameter combinations for experimentation.

    Generated from a ParameterSpace, represents all combinations to test.
    """

    space_name: str
    param_names: list[str]
    fixed_params: dict[str, Any]
    combinations: list[dict[str, Any]]

    def __len__(self) -> int:
        """Number of parameter combinations."""
        return len(self.combinations)

    def __iter__(self):
        """Iterate over parameter combinations."""
        return iter(self.combinations)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get parameter combination by index."""
        return self.combinations[idx]

    def summary(self) -> str:
        """Generate human-readable summary of the grid.

        Returns:
            Summary string describing the parameter space
        """
        lines = [
            f"Parameter Grid: {self.space_name}",
            f"Combinations: {len(self.combinations)}",
            "",
            "Variable parameters:",
        ]

        for param in self.param_names:
            values = {combo[param] for combo in self.combinations}
            if len(values) <= 5:
                lines.append(f"  {param}: {sorted(values)}")
            else:
                sorted_values = sorted(values)
                lines.append(
                    f"  {param}: {sorted_values[0]} ... {sorted_values[-1]} ({len(values)} values)"
                )

        if self.fixed_params:
            lines.append("")
            lines.append("Fixed parameters:")
            for param, value in self.fixed_params.items():
                lines.append(f"  {param}: {value}")

        return "\n".join(lines)
