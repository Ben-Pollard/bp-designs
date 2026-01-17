import json
import os


def create_summary_html(experiment_name):
    exp_dir = f"output/experiments/{experiment_name}"
    outputs_dir = f"{exp_dir}/outputs"

    if not os.path.exists(outputs_dir):
        print(f"Experiment {experiment_name} not found.")
        return

    html = ["<html><head><style>"]
    html.append(".variant { display: inline-block; margin: 10px; border: 1px solid #ccc; padding: 5px; vertical-align: top; }")
    html.append(".variant img { width: 300px; height: 300px; display: block; }")
    html.append(".params { font-size: 10px; max-width: 300px; white-space: pre-wrap; }")
    html.append("</style></head><body>")
    html.append(f"<h1>Experiment: {experiment_name}</h1>")

    # Get all variant JSON files
    variants = sorted([f for f in os.listdir(outputs_dir) if f.endswith('.json') and not f.endswith('_data.json')])

    for v_file in variants:
        with open(os.path.join(outputs_dir, v_file)) as f:
            data = json.load(f)

        v_id = data['variant_id']
        svg_filename = data['svg_path'].split('/')[-1]
        svg_path = os.path.join('outputs', svg_filename)

        # Filter params to show interesting ones
        p = data['params']
        display_params = {
            'seed': p.get('seed'),
            'organ': p.get('organ_type'),
            'scale': p.get('organ_scale'),
            'color': p.get('color_strategy')
        }

        html.append('<div class="variant">')
        html.append(f'<h3>{v_id}</h3>')
        html.append(f'<img src="{svg_path}" />')
        html.append(f'<div class="params">{json.dumps(display_params, indent=2)}</div>')
        html.append('</div>')

    html.append("</body></html>")

    with open(f"{exp_dir}/summary.html", "w") as f:
        f.write("\n".join(html))
    print(f"Summary created at {exp_dir}/summary.html")

if __name__ == "__main__":
    create_summary_html("organic_details_demo")
