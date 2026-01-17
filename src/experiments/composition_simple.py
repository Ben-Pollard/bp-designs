"""Simple composition experiment with just one working composition type."""

from typing import Any

from bp_designs.core.combinator import PatternCombinator
from bp_designs.experiment.params import ParameterSpace
from bp_designs.experiment.runner import ExperimentRunner
from bp_designs.generators.branching.space_colonization import SpaceColonization
from bp_designs.generators.cellular.voronoi import VoronoiTessellation


def generate_textured_example(params: dict[str, Any]) -> dict[str, Any]:
    """Generate a textured composition example.

    Args:
        params: Dictionary of parameters

    Returns:
        Dictionary with 'geometry' and metadata
    """
    seed = params["seed"]

    # Create base patterns
    tree_gen = SpaceColonization(
        seed=seed,
        num_attractions=params.get("num_attractions", 500),
        attraction_distance=params.get("attraction_distance", 50.0),
        segment_length=params.get("segment_length", 2.0),
    )

    voronoi_gen = VoronoiTessellation(
        seed=seed + 1000,  # Different seed for variety
        num_sites=params.get("num_sites", 30),
        relaxation_iterations=params.get("relaxation_iterations", 2),
        render_mode=params.get("render_mode", "edges"),
    )

    voronoi_pattern = voronoi_gen.generate_pattern()
    tree_pattern = tree_gen.generate_pattern()

    # Voronoi texture around tree branches
    pattern = PatternCombinator.texture(
        skeleton=tree_pattern,
        fill=voronoi_pattern,
        distance_threshold=params.get("distance_threshold", 8.0),
    )

    # Convert to geometry
    geometry = pattern.to_geometry()

    return {
        "geometry": geometry,
        "composition_type": "textured",
        "seed": seed,
        "pattern_type": type(pattern).__name__,
    }


def main():
    """Generate simple composition examples."""

    # Define parameter space
    space = ParameterSpace(
        name="composition_simple",
        specs={
            "seed": [42, 123, 456],  # 3 seeds for variety
            "num_attractions": [500],  # Just one value
            "num_sites": [30],  # Just one value
            "distance_threshold": 8.0,
            "attraction_distance": 50.0,
            "segment_length": 2.0,
            "relaxation_iterations": 2,
            "render_mode": "edges",
        },
    )

    # Generate grid
    grid = space.to_grid()
    print(f"Total combinations: {len(grid)}")

    # Run experiment
    runner = ExperimentRunner(
        experiment_name="composition_simple_001",
        svg_width=100,
        svg_height=100,
        stroke_width=0.3,
    )

    runner.run(grid=grid, generator_fn=generate_textured_example)
    print("Simple composition examples generated successfully!")


if __name__ == "__main__":
    main()
