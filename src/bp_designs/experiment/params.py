"""Parameter space definition and grid generation for experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np


@dataclass
class ParameterSpace:
    """Define parameter spaces for systematic experimentation.

    Supports multiple sampling strategies:
    - Linear spacing (linspace)
    - Logarithmic spacing (logspace)
    - Explicit value lists

    Parameters are defined in a unified 'specs' dictionary mapping parameter
    names to specifications:
    - List of explicit values [v1, v2, ...]
    - Tuple (min, max, steps) for linear spacing
    - Single value for fixed parameter

    Namespacing can be achieved using dot-notation in keys (e.g., 'trunk.length').
    """

    name: str
    specs: dict[str, Any]
    derived: dict[str, Callable[[dict[str, Any]], Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate parameter specifications."""
        pass

    def expand_spec(self, param_name: str, spec: Any) -> list[Any]:
        """Expand a parameter specification into a list of values.

        Args:
            param_name: Parameter name (for error messages)
            spec: Either a list of explicit values, tuple (min, max, steps),
                  or a single value (treated as fixed)

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
            # Single value - treat as fixed parameter with one value
            return [spec]

    def to_grid(self) -> ParameterGrid:
        """Generate all parameter combinations as a grid.

        Returns:
            ParameterGrid with all combinations
        """
        # Expand all specifications
        expanded = {}
        for param_name, spec in self.specs.items():
            expanded[param_name] = self.expand_spec(param_name, spec)

        # Generate all combinations of independent parameters
        param_names = list(expanded.keys())
        param_values = [expanded[name] for name in param_names]

        combinations = []
        for values in product(*param_values):
            combo = dict(zip(param_names, values, strict=False))

            # Apply derived parameters
            for derived_name, fn in self.derived.items():
                try:
                    combo[derived_name] = fn(combo)
                except Exception as e:
                    # If a derived parameter fails, we might want to know why
                    # but for now we'll just set it to None or skip
                    combo[derived_name] = f"Error: {e}"

            combinations.append(combo)

        return ParameterGrid(
            space_name=self.name,
            param_names=param_names,
            derived_param_names=list(self.derived.keys()),
            combinations=combinations,
        )


@dataclass
class ParameterGrid:
    """Grid of parameter combinations for experimentation.

    Generated from a ParameterSpace, represents all combinations to test.
    """

    space_name: str
    param_names: list[str]
    combinations: list[dict[str, Any]]
    derived_param_names: list[str] = field(default_factory=list)

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
        ]

        # Combine all parameter names
        all_param_names = self.param_names + self.derived_param_names

        # Separate variable and fixed parameters
        variable_params = []
        fixed_params = []

        for param in all_param_names:
            # Get string representations for uniqueness check (handles unhashable types)
            unique_strings = {str(combo[param]) for combo in self.combinations}
            if len(unique_strings) == 1:
                # All values have same string representation, treat as fixed
                fixed_params.append((param, self.combinations[0][param]))
            else:
                # Variable parameter - store unique values as strings for display
                variable_params.append((param, unique_strings))

        if variable_params:
            lines.append("")
            lines.append("Variable parameters:")
            for param, values in variable_params:
                unique_strings = {str(v) for v in values}
                num_values = len(unique_strings)
                if num_values <= 5:
                    sorted_strings = sorted(unique_strings)
                    lines.append(f"  {param}: {sorted_strings}")
                else:
                    sorted_strings = sorted(unique_strings)
                    lines.append(
                        f"  {param}: {sorted_strings[0]} ... {sorted_strings[-1]} ({num_values} values)"
                    )

        if fixed_params:
            lines.append("")
            lines.append("Fixed parameters:")
            for param, value in fixed_params:
                lines.append(f"  {param}: {value}")

        # Add section headers for categories if needed
        if self.param_names or self.derived_param_names:
            lines.append("")
            lines.append("Parameter categories:")
            if self.param_names:
                lines.append(f"  Independent parameters: {len(self.param_names)}")
            if self.derived_param_names:
                lines.append(f"  Derived parameters: {len(self.derived_param_names)}")

        return "\n".join(lines)


def split_params(params: dict[str, Any], namespace: str | None = None) -> dict[str, Any]:
    """Extract parameters for a specific namespace.

    Args:
        params: Flat dictionary of parameters (potentially namespaced with dots)
        namespace: Namespace to extract (e.g., 'trunk'). If None, returns all
                   non-namespaced parameters.

    Returns:
        Dictionary of parameters with the namespace prefix removed.
    """
    result = {}
    for k, v in params.items():
        if namespace:
            prefix = f"{namespace}."
            if k.startswith(prefix):
                result[k[len(prefix) :]] = v
        else:
            if "." not in k:
                result[k] = v
    return result
