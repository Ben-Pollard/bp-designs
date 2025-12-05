"""Pattern composition operations.

High-level composition operations that work on any Pattern instances
via their field interface. These enable universal pattern combination
without custom pairwise logic.
"""

from __future__ import annotations

import numpy as np

from bp_designs.core.generator import Generator
from bp_designs.core.pattern import Pattern

from .composite_pattern import CompositePattern


class PatternCombinator:
    """High-level composition operations between patterns.

    This is the "algebra" - semantic operations that work on any Pattern
    instances via their field interface. These preserve the compositional
    goal from Options 2 & 3 in the original exploration.

    Key Insight:
        Patterns maintain semantic structures internally, but expose
        themselves through fields. Combinators use field queries to
        implement high-level operations without knowing internal details.

    Combinators provide:
    - Semantic vocabulary (guide, texture, nest, blend)
    - Universal operations (work on any Pattern pair)
    - Recursive composition (results are also Patterns)
    """

    @staticmethod
    def guide(
        structure: Pattern,
        generator: Generator,
        influence_channel: str,
        influence_strength: float = 1.0,
        **generator_kwargs,
    ) -> Pattern:
        """Let structure influence pattern generation via field.

        Semantic Operation:
            One pattern provides a "field of influence" that guides
            how another pattern grows.

        Args:
            structure: Pattern providing guidance field (e.g., Voronoi cells)
            generator: Generator to be influenced (e.g., SpaceColonization)
            influence_channel: Which channel from structure to use
            influence_strength: Scaling factor for influence
            **generator_kwargs: Additional args passed to generator

        Returns:
            Generated pattern (type depends on generator)

        Example:
            # Voronoi cell boundaries guide tree growth
            voronoi = VoronoiTessellation(...).generate_pattern()
            tree_gen = SpaceColonization(...)

            guided_tree = PatternCombinator.guide(
                structure=voronoi,
                generator=tree_gen,
                influence_channel='boundary_distance',
                influence_strength=0.5
            )

        Implementation:
            Simply passes structure.sample_field to the generator.
            The generator uses field queries during its growth process.
        """
        return generator.generate_pattern(
            guidance_field=structure.sample_field,
            guidance_channel=influence_channel,
            guidance_strength=influence_strength,
            **generator_kwargs,
        )

    @staticmethod
    def texture(
        skeleton: Pattern, fill: Pattern, distance_threshold: float, fill_channel: str = "distance"
    ) -> CompositePattern:
        """Fill skeleton structure with texture pattern.

        Semantic Operation:
            One pattern defines structure (skeleton), another provides
            texture that fills regions near the skeleton.

        Args:
            skeleton: Pattern defining structure (e.g., tree branches)
            fill: Pattern providing texture (e.g., Voronoi cells)
            distance_threshold: Maximum distance from skeleton to fill
            fill_channel: Channel to use for distance queries

        Returns:
            CompositePattern combining both

        Example:
            # Cellular texture around tree branches
            tree = SpaceColonization(...).generate_pattern()
            voronoi = VoronoiTessellation(...).generate_pattern()

            textured = PatternCombinator.texture(
                skeleton=tree,
                fill=voronoi,
                distance_threshold=5.0
            )

        Implementation:
            Queries skeleton distance field to create a spatial mask.
            Fill geometry is clipped to regions within threshold.
        """

        def combinator(points, channel, components, metadata):
            skeleton, fill = components
            threshold = metadata["distance_threshold"]

            if channel.startswith("skeleton_"):
                return skeleton.sample_field(points, channel[9:])
            elif channel.startswith("fill_"):
                return fill.sample_field(points, channel[5:])
            elif channel == "mask":
                return skeleton.sample_field(points, "distance") < threshold
            else:
                raise ValueError(f"Unknown channel: {channel}")

        return CompositePattern(
            components=[skeleton, fill],
            combinator=combinator,
            metadata={
                "type": "texture",
                "distance_threshold": distance_threshold,
                "fill_channel": fill_channel,
            },
        )

    @staticmethod
    def nest(
        container: Pattern,
        content_generator: Generator,
        region_channel: str = "cell_id",
        **generator_kwargs,
    ) -> CompositePattern:
        """Generate content pattern within regions of container.

        Semantic Operation:
            One pattern defines spatial regions (container), content
            is generated independently within each region.

        Args:
            container: Pattern defining regions (e.g., Voronoi cells)
            content_generator: Generator to run in each region
            region_channel: Channel that identifies distinct regions
            **generator_kwargs: Args for content generator

        Returns:
            CompositePattern with content in each region

        Example:
            # Small tree in each Voronoi cell
            voronoi = VoronoiTessellation(...).generate_pattern()
            tree_gen = SpaceColonization(bounds=(0,0,10,10), ...)

            trees_in_cells = PatternCombinator.nest(
                container=voronoi,
                content_generator=tree_gen,
                region_channel='cell_id'
            )

        Implementation:
            1. Query container to identify distinct regions via field
            2. For each region, generate content with appropriate bounds
            3. Combine all generated content
        """
        # Extract regions from container
        # This requires pattern-specific logic or a standard approach
        # For now, show the interface

        def combinator(points, channel, components, metadata):
            container = components[0]
            components[1:]

            # Delegate to appropriate component
            if channel.startswith("container_"):
                return container.sample_field(points, channel[10:])
            elif channel.startswith("content_"):
                # Would need to route to correct content based on region
                raise NotImplementedError("Region-based routing")
            else:
                raise ValueError(f"Unknown channel: {channel}")

        # Placeholder - actual implementation needs region extraction
        contents = []  # Generate content in each region

        return CompositePattern(
            components=[container] + contents,
            combinator=combinator,
            metadata={"type": "nest", "region_channel": region_channel, "n_regions": len(contents)},
        )

    @staticmethod
    def blend(
        pattern_a: Pattern,
        pattern_b: Pattern,
        blend_mode: str,  # 'multiply', 'add', 'mask', 'max', 'min'
        channel_a: str = "density",
        channel_b: str = "density",
    ) -> CompositePattern:
        """Blend two patterns using field operations.

        Semantic Operation:
            Two patterns interact via field-based arithmetic operations.

        Args:
            pattern_a: First pattern
            pattern_b: Second pattern
            blend_mode: How to combine field values
            channel_a: Channel to sample from pattern_a
            channel_b: Channel to sample from pattern_b

        Returns:
            CompositePattern with blended fields

        Example:
            # Multiply tree density with cell boundaries
            tree = SpaceColonization(...).generate_pattern()
            voronoi = VoronoiTessellation(...).generate_pattern()

            blended = PatternCombinator.blend(
                pattern_a=tree,
                pattern_b=voronoi,
                blend_mode='multiply',
                channel_a='density',
                channel_b='boundary_distance'
            )

        Implementation:
            Sample both patterns at query points, apply arithmetic operation.
        """

        def combinator(points, channel, components, metadata):
            pattern_a, pattern_b = components
            mode = metadata["mode"]
            ch_a = metadata["channel_a"]
            ch_b = metadata["channel_b"]

            if channel == "blended":
                values_a = pattern_a.sample_field(points, ch_a)
                values_b = pattern_b.sample_field(points, ch_b)

                if mode == "multiply":
                    return values_a * values_b
                elif mode == "add":
                    return values_a + values_b
                elif mode == "max":
                    return np.maximum(values_a, values_b)
                elif mode == "min":
                    return np.minimum(values_a, values_b)
                elif mode == "mask":
                    return values_a * (values_b > 0.5)
                else:
                    raise ValueError(f"Unknown blend mode: {mode}")

            elif channel.startswith("a_"):
                return pattern_a.sample_field(points, channel[2:])
            elif channel.startswith("b_"):
                return pattern_b.sample_field(points, channel[2:])
            else:
                raise ValueError(f"Unknown channel: {channel}")

        return CompositePattern(
            components=[pattern_a, pattern_b],
            combinator=combinator,
            metadata={
                "type": "blend",
                "mode": blend_mode,
                "channel_a": channel_a,
                "channel_b": channel_b,
            },
        )
