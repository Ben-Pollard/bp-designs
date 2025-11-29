"""SVG export for geometry with Jupyter support."""

import svgwrite

from bp_designs.geometry import Geometry


def geometry_to_svg(
    geometry: Geometry,
    width: float = 100,
    height: float = 100,
    stroke_width: float = 0.5,
    stroke_color: str = "#000000",
    background: str | None = "#ffffff",
) -> str:
    """Convert geometry to SVG string.

    Args:
        geometry: List of polylines to render
        width: SVG width in mm
        height: SVG height in mm
        stroke_width: Line width in mm
        stroke_color: Stroke color (hex)
        background: Background color (hex) or None for transparent

    Returns:
        SVG string
    """
    # Create SVG with viewBox for proper scaling
    dwg = svgwrite.Drawing(size=(f"{width}mm", f"{height}mm"), viewBox=f"0 0 {width} {height}")

    # Add background if specified
    if background:
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill=background))

    # Add each polyline
    for polyline in geometry:
        if len(polyline) < 2:
            continue

        points = [(float(p[0]), float(p[1])) for p in polyline]
        dwg.add(
            dwg.polyline(
                points=points,
                stroke=stroke_color,
                stroke_width=stroke_width,
                fill="none",
                stroke_linecap="round",
                stroke_linejoin="round",
            )
        )

    return dwg.tostring()


def render_svg(
    geometry: Geometry,
    width: float = 100,
    height: float = 100,
    stroke_width: float = 0.5,
    stroke_color: str = "#000000",
    background: str | None = "#ffffff",
):
    """Render geometry as SVG for Jupyter display.

    This function returns an object that Jupyter will automatically
    display as an SVG image.

    Args:
        geometry: List of polylines to render
        width: SVG width in mm
        height: SVG height in mm
        stroke_width: Line width in mm
        stroke_color: Stroke color (hex)
        background: Background color (hex) or None for transparent

    Returns:
        IPython.display.SVG object (if in Jupyter) or SVG string
    """
    svg_string = geometry_to_svg(
        geometry,
        width=width,
        height=height,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background=background,
    )

    # Try to use IPython display if available
    try:
        from IPython.display import SVG

        return SVG(data=svg_string)
    except ImportError:
        # Not in Jupyter, just return string
        return svg_string
