"""Generate composition examples for gallery demonstration."""

from typing import Any

from bp_designs.core.combinator import PatternCombinator
from bp_designs.experiment import ExperimentRunner, ParameterSpace
from bp_designs.patterns.branching.space_colonization import SpaceColonization
from bp_designs.patterns.cellular.voronoi import VoronoiTessellation


def generate_composition_example(params: dict[str, Any]) -> dict[str, Any]:
    """Generate a single composition example.

    Args:
        params: Dictionary of parameters

    Returns:
        Dictionary with 'geometry' and metadata
    """
    composition_type = params["composition_type"]
    seed = params["seed"]

    # Create base patterns
    tree_gen = SpaceColonization(
        seed=seed,
        num_attractions=params.get("num_attractions", 150),
        attraction_distance=params.get("attraction_distance", 30.0),
        segment_length=params.get("segment_length", 2.0),
    )

    voronoi_gen = VoronoiTessellation(
        seed=seed + 1000,  # Different seed for variety
        num_sites=params.get("num_sites", 40),
        relaxation_iterations=params.get("relaxation_iterations", 2),
        render_mode=params.get("render_mode", "edges"),
    )

    voronoi_pattern = voronoi_gen.generate()

    if composition_type == "guided":
        # Tree growth guided by Voronoi boundaries
        pattern = PatternCombinator.guide(
            structure=voronoi_pattern,
            generator=tree_gen,
            influence_channel=params.get("influence_channel", "boundary_distance"),
            influence_strength=params.get("influence_strength", 0.5),
        )

    elif composition_type == "textured":
        # Voronoi texture around tree branches
        tree_pattern = tree_gen.generate_network()
        pattern = PatternCombinator.texture(
            skeleton=tree_pattern,
            fill=voronoi_pattern,
            distance_threshold=params.get("distance_threshold", 8.0),
        )

    elif composition_type == "blended":
        # Blend tree density with Voronoi boundaries
        tree_pattern = tree_gen.generate_network()
        pattern = PatternCombinator.blend(
            pattern_a=tree_pattern,
            pattern_b=voronoi_pattern,
            blend_mode=params.get("blend_mode", "multiply"),
            channel_a=params.get("channel_a", "density"),
            channel_b=params.get("channel_b", "boundary_distance"),
        )

    else:
        raise ValueError(f"Unknown composition type: {composition_type}")

    # Convert to geometry
    geometry = pattern.to_geometry()

    return {
        "geometry": geometry,
        "composition_type": composition_type,
        "seed": seed,
        "pattern_type": type(pattern).__name__,
    }


def main():
    """Generate composition examples."""

    # Define parameter space
    space = ParameterSpace(
        name="composition_examples",
        ranges={
            "composition_type": ["guided", "textured", "blended"],
            "seed": [42, 123, 456],  # 3 seeds for variety
            "num_attractions": [500, 1000],  # 2 values - higher for better growth
            "num_sites": [30, 40],  # 2 values
        },
        fixed={
            "influence_strength": 0.5,
            "distance_threshold": 8.0,
            "blend_mode": "multiply",
            "attraction_distance": 50.0,  # Higher attraction distance
            "segment_length": 2.0,
            "relaxation_iterations": 2,
            "render_mode": "edges",
            "influence_channel": "boundary_distance",
            "channel_a": "density",
            "channel_b": "boundary_distance",
        },
    )

    # Generate grid
    grid = space.to_grid()
    print(f"Total combinations: {len(grid)}")

    # Run experiment
    runner = ExperimentRunner(
        experiment_name="composition_examples_001",
        svg_width=100,
        svg_height=100,
        stroke_width=0.3,
    )

    runner.run(grid=grid, generator_fn=generate_composition_example)
    print("Composition examples generated successfully!")


if __name__ == "__main__":
    main()
