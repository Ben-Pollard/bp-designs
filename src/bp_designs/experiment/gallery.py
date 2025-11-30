"""Gallery HTML generator for experiment visualization."""

from __future__ import annotations

import json
from pathlib import Path


def generate_gallery(experiment_dir: str | Path, title: str | None = None) -> Path:
    """Generate interactive HTML gallery from experiment outputs.

    Args:
        experiment_dir: Path to experiment directory (contains outputs/)
        title: Optional custom title (defaults to experiment name)

    Returns:
        Path to generated gallery.html file
    """
    exp_dir = Path(experiment_dir)
    outputs_dir = exp_dir / "outputs"

    if not outputs_dir.exists():
        raise ValueError(f"Outputs directory not found: {outputs_dir}")

    # Load experiment config
    config_path = exp_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"experiment_name": exp_dir.name}

    # Collect all variants
    variants = []
    for json_path in sorted(outputs_dir.glob("var_*.json")):
        with open(json_path) as f:
            metadata = json.load(f)
            variants.append(metadata)

    # Generate HTML
    gallery_title = title or config.get("experiment_name", "Experiment Gallery")
    html = _generate_html(gallery_title, variants, config)

    # Save gallery
    gallery_path = exp_dir / "gallery.html"
    gallery_path.write_text(html)

    print(f"Gallery generated: {gallery_path}")
    print(f"Variants: {len(variants)}")

    return gallery_path


def _generate_html(title: str, variants: list[dict], config: dict) -> str:
    """Generate HTML content for gallery.

    Args:
        title: Gallery title
        variants: List of variant metadata dicts
        config: Experiment config

    Returns:
        HTML string
    """
    # Generate variant cards
    cards_html = []
    for variant in variants:
        params_html = "<br>".join(
            f"<strong>{k}:</strong> {_format_value(v)}" for k, v in variant["params"].items()
        )

        card = f"""
        <div class="card">
            <div class="image-container">
                <img src="{variant["svg_path"]}" alt="{variant["variant_id"]}">
            </div>
            <div class="metadata">
                <div class="variant-id">{variant["variant_id"]}</div>
                <div class="params">{params_html}</div>
            </div>
        </div>
        """
        cards_html.append(card)

    # Experiment summary
    summary_html = f"""
    <div class="summary">
        <strong>Variants:</strong> {config.get("total_variants", len(variants))} |
        <strong>Successful:</strong> {config.get("successful", len(variants))} |
        <strong>Failed:</strong> {config.get("failed", 0)}
    </div>
    """

    # Full HTML document
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
            Oxygen, Ubuntu, sans-serif;
            background: #f5f5f5;
            padding: 20px;
            line-height: 1.6;
        }}

        .header {{
            background: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}

        .summary {{
            color: #666;
            font-size: 14px;
        }}

        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        .card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        .image-container {{
            width: 100%;
            height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #fafafa;
            border-bottom: 1px solid #eee;
        }}

        .image-container img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}

        .metadata {{
            padding: 15px;
        }}

        .variant-id {{
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .params {{
            font-size: 12px;
            color: #666;
            line-height: 1.8;
        }}

        .params strong {{
            color: #333;
        }}

        .footer {{
            text-align: center;
            color: #999;
            font-size: 12px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        {summary_html}
    </div>

    <div class="gallery">
        {"".join(cards_html)}
    </div>

    <div class="footer">
        Generated by BP Designs Experimentation Framework
    </div>
</body>
</html>
"""  # noqa: E501

    return html


def _format_value(value: any) -> str:
    """Format parameter value for display.

    Args:
        value: Parameter value

    Returns:
        Formatted string
    """
    if isinstance(value, float):
        # Format floats nicely
        return f"{value:.4g}"
    return str(value)
