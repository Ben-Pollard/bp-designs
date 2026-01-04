"""Parameter space definition and grid generation for experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np


@dataclass
class ParameterSpace:
    """Define parameter spaces for pattern generation and rendering.

    Supports multiple sampling strategies:
    - Linear spacing (linspace)
    - Logarithmic spacing (logspace)
    - Explicit value lists

    Parameters are categorized as pattern (for pattern generation) or
    render (for SVG rendering). Each category is a dict mapping parameter
    names to specifications:
    - List of explicit values [v1, v2, ...]
    - Tuple (min, max, steps) for linear spacing
    - Single value for fixed parameter
    """

    name: str
    pattern: dict[str, Any]
    render: dict[str, Any]

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
        # Expand pattern parameters
        expanded_pattern = {}
        for param_name, spec in self.pattern.items():
            expanded_pattern[param_name] = self.expand_spec(param_name, spec)

        # Expand render parameters
        expanded_render = {}
        for param_name, spec in self.render.items():
            expanded_render[param_name] = self.expand_spec(param_name, spec)

        # Generate pattern combinations
        pattern_names = list(expanded_pattern.keys())
        pattern_values = [expanded_pattern[name] for name in pattern_names]
        pattern_combinations = []
        for values in product(*pattern_values):
            pattern_combinations.append(dict(zip(pattern_names, values, strict=False)))

        # Generate render combinations
        render_names = list(expanded_render.keys())
        render_values = [expanded_render[name] for name in render_names]
        render_combinations = []
        for values in product(*render_values):
            render_combinations.append(dict(zip(render_names, values, strict=False)))

        # Cross product: combine each pattern combo with each render combo
        combinations = []
        for pattern_combo in pattern_combinations:
            for render_combo in render_combinations:
                combo = {**pattern_combo, **render_combo}
                combinations.append(combo)

        return ParameterGrid(
            space_name=self.name,
            pattern_param_names=pattern_names,
            render_param_names=render_names,
            combinations=combinations,
        )


@dataclass
class ParameterGrid:
    """Grid of parameter combinations for experimentation.

    Generated from a ParameterSpace, represents all combinations to test.
    """

    space_name: str
    pattern_param_names: list[str]
    render_param_names: list[str]
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
        ]

        # Combine all parameter names
        all_param_names = self.pattern_param_names + self.render_param_names

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

        # Add section headers for pattern vs render if needed
        if self.pattern_param_names and self.render_param_names:
            lines.append("")
            lines.append("Parameter categories:")
            lines.append(f"  Pattern parameters: {len(self.pattern_param_names)}")
            lines.append(f"  Render parameters: {len(self.render_param_names)}")

        return "\n".join(lines)
