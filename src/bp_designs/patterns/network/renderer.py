"""Rendering logic for branching networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bp_designs.core.color import Color
from bp_designs.core.geometry import Canvas, Polygon, Polyline
from bp_designs.core.renderer import RenderStyle
from bp_designs.patterns.network.color import ColorStrategy
from bp_designs.patterns.network.thickness import BranchThicknessStrategy

if TYPE_CHECKING:
    from bp_designs.core.renderer import RenderingContext
    from bp_designs.patterns.network.base import BranchNetwork


class NetworkStyle(RenderStyle):
    """Structured rendering parameters for branching networks."""

    render_mode: str = "polyline"
    thickness: str = "descendant"
    min_thickness: float = 0.5
    max_thickness: float = 5.0
    taper_power: float = 0.5
    thickness_mode: str = "all_nodes"
    taper_style: str = "smooth"
    color: str | Color | None = None
    color_strategy: str | None = None
    stroke_linecap: str = "round"
    stroke_linejoin: str = "round"
    render_branches: bool = True
    render_organs: bool = True
    shading: str | None = None
    organ_color_override: bool = False
    debug_organs: bool = False
    start_color: str | Color = "#4a2c2a"
    end_color: str | Color = "#2d5a27"


class NetworkRenderer:
    """Handles conversion of BranchNetwork to various geometric and visual formats."""

    def __init__(self, network: BranchNetwork):
        self.network = network

    def to_geometry(self, canvas: Canvas | None = None) -> Polyline:
        """Convert to polyline representation for export."""
        positions = self.network.positions
        if canvas is not None and self.network.pattern_bounds is not None:
            orig_xmin, orig_ymin, orig_xmax, orig_ymax = self.network.pattern_bounds
            orig_width = orig_xmax - orig_xmin
            orig_height = orig_ymax - orig_ymin
            new_xmin, new_ymin, new_xmax, new_ymax = canvas.bounds()
            new_width = new_xmax - new_xmin
            new_height = new_ymax - new_ymin
            if orig_width > 0 and orig_height > 0:
                positions = positions.copy()
                positions[:, 0] = (positions[:, 0] - orig_xmin) / orig_width
                positions[:, 1] = (positions[:, 1] - orig_ymin) / orig_height
                positions[:, 0] = new_xmin + positions[:, 0] * new_width
                positions[:, 1] = new_ymin + positions[:, 1] * new_height

        leaves = self.network.get_leaves()
        branches = []
        id_to_idx = {node_id: i for i, node_id in enumerate(self.network.node_ids)}
        for leaf_idx in leaves:
            path = []
            curr_idx = leaf_idx
            while curr_idx != -1:
                path.append(positions[curr_idx])
                parent_id = self.network.parents[curr_idx]
                if parent_id == -1:
                    curr_idx = -1
                else:
                    curr_idx = id_to_idx.get(parent_id, -1)
            if len(path) >= 2:
                branches.append(np.array(path[::-1]))
        return Polyline(polylines=branches)

    def to_polygon(
        self,
        thickness: str = "descendant",
        min_thickness: float = 0.5,
        max_thickness: float = 5.0,
        taper_power: float = 0.5,
        thickness_mode: str = "all_nodes",
        **kwargs,
    ) -> list[Polygon]:
        """Convert network to a single unioned polygon skin."""
        from shapely.geometry import Point as ShapelyPoint

        strategy = BranchThicknessStrategy.from_name(
            thickness,
            min_thickness=min_thickness,
            max_thickness=max_thickness,
            power=taper_power,
            mode=thickness_mode,
        )
        all_thickness = strategy.compute_thickness(self.network)

        import shapely

        # 1. Pre-calculate all node circles
        node_circles = []
        for i in range(len(self.network.node_ids)):
            pos = self.network.positions[i]
            r = all_thickness[i] / 2.0
            node_circles.append(ShapelyPoint(pos).buffer(r))

        # 2. Create envelopes for segments
        shapes = list(node_circles)
        id_to_idx = {node_id: i for i, node_id in enumerate(self.network.node_ids)}
        for i in range(len(self.network.node_ids)):
            parent_id = self.network.parents[i]
            if parent_id == -1 or parent_id not in id_to_idx:
                continue

            parent_idx = id_to_idx[parent_id]
            # The convex hull of two circles is the trapezoidal envelope connecting them.
            # Using GeometryCollection avoids the expensive unary_union inside the loop.
            envelope = shapely.convex_hull(
                shapely.GeometryCollection([node_circles[i], node_circles[parent_idx]])
            )
            shapes.append(envelope)

        # 3. Perform a single union of all shapes at the end
        combined = shapely.union_all(shapes)
        polygons = []
        if combined.is_empty:
            return []
        if combined.geom_type == "Polygon":
            polygons.append(Polygon(coords=np.array(combined.exterior.coords)))
        elif combined.geom_type == "MultiPolygon":
            for poly in combined.geoms:
                polygons.append(Polygon(coords=np.array(poly.exterior.coords)))
        return polygons

    def render(
        self,
        context: RenderingContext,
        style: NetworkStyle | None = None,
        **kwargs,
    ):
        """Render branch network into the provided context."""
        # Prioritize parameters:
        # 1. Provided style object
        # 2. Stored render_params on the network
        # 3. kwargs passed to this method (global context)

        # Start with stored params
        params = self.network.render_params.copy()
        # Update with kwargs (overrides/global context)
        params.update(kwargs)

        # Merge style and params
        if style is None:
            style = NetworkStyle.from_dict(params)
        else:
            # Allow params to override style fields
            style_dict = style.model_dump()
            style_dict.update(params)
            style = NetworkStyle.from_dict(style_dict)

        # Always use strategy for thickness unless explicitly overridden (not implemented yet)
        strategy = BranchThicknessStrategy.from_name(
            style.thickness,
            min_thickness=style.min_thickness,
            max_thickness=style.max_thickness,
            power=style.taper_power,
            mode=style.thickness_mode,
        )
        all_thickness = strategy.compute_thickness(self.network)

        # Always use strategy for color if provided
        if style.color_strategy is not None:
            strategy = ColorStrategy.from_name(
                style.color_strategy,
                start_color=style.start_color,
                end_color=style.end_color,
            )
            all_colors = strategy.compute_colors(self.network)
        else:
            default_color = style.color if style.color is not None else "#000000"
            all_colors = np.full(len(self.network.node_ids), default_color, dtype=object)

        svg_attrs = style.get_svg_attributes()

        id_to_idx = {node_id: i for i, node_id in enumerate(self.network.node_ids)}

        # Render branches
        if style.render_branches:
            context.push_group("branches")
            if style.render_mode == "polygon":
                polygons = self.to_polygon(
                    thickness=style.thickness,
                    min_thickness=style.min_thickness,
                    max_thickness=style.max_thickness,
                    taper_power=style.taper_power,
                    thickness_mode=style.thickness_mode,
                )
                for poly in polygons:
                    # Use the first node's color as base for the unioned polygon
                    fill_color = all_colors[0] if len(all_colors) > 0 else "black"

                    # Use lighting from context if available, otherwise check kwargs
                    lighting = getattr(context, "lighting", None) or kwargs.get("lighting")

                    if lighting:
                        # For unioned polygons, we use a global directional fill
                        fill_color = lighting.get_fill(
                            fill_color, {"type": "global"}
                        )
                    context.add(
                        context.dwg.polygon(
                            points=[(float(x), float(y)) for x, y in poly.coords],
                            fill=str(fill_color),
                            **svg_attrs,
                        )
                    )
            else:
                for i in range(len(self.network.node_ids)):
                    node_color = all_colors[i]
                    parent_id = self.network.parents[i]
                    if parent_id == -1 or parent_id not in id_to_idx:
                        continue
                    parent_idx = id_to_idx[parent_id]
                    start = self.network.positions[parent_idx]
                    end = self.network.positions[i]
                    t_start = all_thickness[parent_idx]
                    t_end = all_thickness[i]

                    if style.taper_style == "smooth":
                        v = end - start
                        length = np.linalg.norm(v)
                        if length > 0:
                            n = np.array([-v[1], v[0]]) / length
                            p1 = start + n * (t_start / 2)
                            p2 = start - n * (t_start / 2)
                            p3 = end - n * (t_end / 2)
                            p4 = end + n * (t_end / 2)
                            fill = node_color
                            if context.lighting:
                                fill = context.lighting.get_fill(
                                    node_color, {"type": "branch", "vector": v}
                                )
                            context.add(
                                context.dwg.polygon(
                                    points=[
                                        (float(p1[0]), float(p1[1])),
                                        (float(p2[0]), float(p2[1])),
                                        (float(p3[0]), float(p3[1])),
                                        (float(p4[0]), float(p4[1])),
                                    ],
                                    fill=str(fill),
                                    **svg_attrs,
                                )
                            )
                            context.add(
                                context.dwg.circle(
                                    center=(float(start[0]), float(start[1])),
                                    r=float(t_start / 2),
                                    fill=str(fill),
                                    **svg_attrs,
                                )
                            )
                            context.add(
                                context.dwg.circle(
                                    center=(float(end[0]), float(end[1])),
                                    r=float(t_end / 2),
                                    fill=str(fill),
                                    **svg_attrs,
                                )
                            )
                    elif style.taper_style == "blocky":
                        context.add(
                            context.dwg.line(
                                start=(float(start[0]), float(start[1])),
                                end=(float(end[0]), float(end[1])),
                                stroke=str(node_color),
                                stroke_width=all_thickness[i],
                                stroke_linecap=style.stroke_linecap,
                                stroke_linejoin=style.stroke_linejoin,
                                **svg_attrs,
                            )
                        )
            context.pop_group()

        # Render organs
        if style.render_organs and self.network.organs:
            context.push_group("organs")
            for node_id, organ in self.network.organs.items():
                idx = id_to_idx.get(node_id)
                if idx is not None:
                    angle = 0.0
                    parent_id = self.network.parents[idx]
                    if parent_id != -1:
                        parent_idx = id_to_idx.get(parent_id)
                        if parent_idx is not None:
                            v = self.network.positions[idx] - self.network.positions[parent_idx]
                            angle = np.degrees(np.arctan2(v[1], v[0]))
                    scale_factor = (
                        all_thickness[idx] / style.max_thickness if style.max_thickness > 0 else 1.0
                    )
                    # Ensure scale factor doesn't make organs too small to see
                    scale_factor = max(scale_factor, 0.5)
                    # Use branch color only if override is requested, otherwise let organ use its base_color
                    organ_color = all_colors[idx] if style.organ_color_override else None
                    organ.render(
                        context,
                        self.network.positions[idx],
                        color=organ_color,
                        orientation=angle,
                        scale_factor=scale_factor,
                    )
                    if style.debug_organs:
                        # Draw a bright red circle at the attachment point
                        context.add(
                            context.dwg.circle(
                                center=(
                                    float(self.network.positions[idx][0]),
                                    float(self.network.positions[idx][1]),
                                ),
                                r=2.0,
                                fill="red",
                                fill_opacity=0.5,
                                stroke="white",
                                stroke_width=0.5,
                            )
                        )
            context.pop_group()
