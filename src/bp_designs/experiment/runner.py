"""Experiment runner for systematic pattern generation."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bp_designs.export.svg import geometry_to_svg

if TYPE_CHECKING:
    from bp_designs.experiment.params import ParameterGrid
    from bp_designs.geometry import Geometry


class ExperimentRunner:
    """Run parameter sweep experiments and save results.

    Generates pattern variants across a parameter grid, saving SVG outputs
    and metadata for gallery display.
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str | Path = "experiments",
        svg_width: float = 100,
        svg_height: float = 100,
        stroke_width: float = 0.3,
    ):
        """Initialize experiment runner.

        Args:
            experiment_name: Name for this experiment (used as directory name)
            output_dir: Base directory for all experiments
            svg_width: SVG canvas width in mm
            svg_height: SVG canvas height in mm
            stroke_width: Default stroke width in mm
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.svg_width = svg_width
        self.svg_height = svg_height
        self.stroke_width = stroke_width

        # Create experiment directory structure
        self.exp_dir = self.output_dir / experiment_name
        self.outputs_dir = self.exp_dir / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        grid: ParameterGrid,
        generator_fn: Callable[[dict[str, Any]], Geometry],
        max_variants: int | None = None,
    ) -> dict[str, Any]:
        """Run experiment across parameter grid.

        Args:
            grid: ParameterGrid defining combinations to test
            generator_fn: Function that takes params dict and returns Geometry
            max_variants: Optional limit on number of variants (for testing)

        Returns:
            Experiment summary with metadata
        """
        print(f"\n{'=' * 60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"{'=' * 60}")
        print(grid.summary())
        print(f"\n{'=' * 60}\n")

        # Limit variants if requested
        num_variants = len(grid) if max_variants is None else min(len(grid), max_variants)

        start_time = time.time()
        successes = 0
        failures = []

        for i, params in enumerate(grid):
            if i >= num_variants:
                break

            variant_id = f"var_{i + 1:04d}"
            print(f"[{i + 1}/{num_variants}] Generating {variant_id}...", end=" ")

            try:
                # Generate pattern
                geometry = generator_fn(params)

                # Save SVG
                svg_path = self.outputs_dir / f"{variant_id}.svg"
                svg_string = geometry_to_svg(
                    geometry,
                    width=self.svg_width,
                    height=self.svg_height,
                    stroke_width=self.stroke_width,
                )
                svg_path.write_text(svg_string)

                # Save metadata
                metadata = {
                    "variant_id": variant_id,
                    "params": params,
                    "svg_path": f"outputs/{variant_id}.svg",
                    "svg_size": {"width": self.svg_width, "height": self.svg_height},
                    "stroke_width": self.stroke_width,
                }
                metadata_path = self.outputs_dir / f"{variant_id}.json"
                metadata_path.write_text(json.dumps(metadata, indent=2))

                print("✓")
                successes += 1

            except Exception as e:
                print(f"✗ ({e})")
                failures.append({"variant_id": variant_id, "params": params, "error": str(e)})

        elapsed = time.time() - start_time

        # Save experiment config and summary
        experiment_summary = {
            "experiment_name": self.experiment_name,
            "parameter_space": grid.space_name,
            "total_variants": num_variants,
            "successful": successes,
            "failed": len(failures),
            "failures": failures,
            "elapsed_seconds": elapsed,
            "svg_config": {
                "width": self.svg_width,
                "height": self.svg_height,
                "stroke_width": self.stroke_width,
            },
        }

        config_path = self.exp_dir / "config.json"
        config_path.write_text(json.dumps(experiment_summary, indent=2))

        print(f"\n{'=' * 60}")
        print(f"Complete: {successes} successful, {len(failures)} failed")
        print(f"Time: {elapsed:.1f}s ({elapsed / num_variants:.2f}s per variant)")
        print(f"Output: {self.exp_dir}")
        print(f"{'=' * 60}\n")

        return experiment_summary
