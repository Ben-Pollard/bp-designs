import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def analyze_variant(variant_id, experiment_name="space_col_oval_variants"):
    data_path = Path(f"output/experiments/{experiment_name}/outputs/{variant_id}_data.json")
    meta_path = Path(f"output/experiments/{experiment_name}/outputs/{variant_id}.json")

    if not data_path.exists():
        print(f"Data file {data_path} not found")
        return

    with open(data_path) as f:
        data = json.load(f)

    with open(meta_path) as f:
        meta = json.load(f)

    parents = np.array(data["parents"])
    timestamps = np.array(data["timestamps"])
    node_ids = np.array(data["node_ids"])
    num_nodes = len(node_ids)

    # Build adjacency list for children
    children_map = defaultdict(list)
    for i, p_id in enumerate(parents):
        if p_id != -1:
            children_map[p_id].append(i)

    # Recompute descendant counts as done in DescendantThickness
    id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    mode = meta["params"].get("thickness_mode", "all_nodes")
    if mode == "leaves_only":
        descendant_counts = np.zeros(num_nodes, dtype=int)
        # A node is a leaf if it's not a parent
        is_parent = np.zeros(num_nodes, dtype=bool)
        for p_id in parents:
            if p_id != -1 and p_id in id_to_idx:
                is_parent[id_to_idx[p_id]] = True
        descendant_counts[~is_parent] = 1
    else:
        descendant_counts = np.ones(num_nodes, dtype=int)

    sorted_indices = np.argsort(timestamps)[::-1]

    for idx in sorted_indices:
        parent_id = parents[idx]
        if parent_id >= 0:
            if parent_id in id_to_idx:
                parent_idx = id_to_idx[parent_id]
                descendant_counts[parent_idx] += descendant_counts[idx]

    max_count = descendant_counts.max()
    power = meta["params"].get("taper_power", 0.5)
    min_t = meta["params"].get("min_thickness", 0.5)
    max_t = meta["params"].get("max_thickness", 5.0)

    normalized = (descendant_counts / max_count) ** power
    thickness = min_t + normalized * (max_t - min_t)

    print(f"\n{'='*20} Analysis for {variant_id} {'='*20}")
    print(f"Num nodes: {num_nodes}")
    print(f"Max descendant count: {max_count}")
    print(f"Taper power: {power}")
    print(f"Thickness range: {thickness.min():.2f} to {thickness.max():.2f}")

    # Find root
    root_indices = np.where(parents == -1)[0]
    if len(root_indices) == 0:
        print("No root found")
        return
    root_idx = root_indices[0]

    def print_tree(idx, depth=0, max_depth=10):
        if depth > max_depth:
            return

        node_id = node_ids[idx]
        children = children_map.get(node_id, [])

        indent = "  " * depth
        branch_symbol = "└── " if depth > 0 else ""

        info = f"ID:{node_id} | Desc:{descendant_counts[idx]} | Thick:{thickness[idx]:.2f}"
        print(f"{indent}{branch_symbol}{info}")

        for child_idx in children:
            print_tree(child_idx, depth + 1, max_depth)

    print("\nTopology (first 10 levels):")
    print_tree(root_idx)

    # Analyze junctions
    print("\nJunction Analysis (Top 5 by thickness jump):")
    junction_jumps = []
    for i in range(num_nodes):
        node_id = node_ids[i]
        children = children_map.get(node_id, [])
        if len(children) > 1:
            p_thick = thickness[i]
            for c_idx in children:
                c_thick = thickness[c_idx]
                jump = p_thick - c_thick
                junction_jumps.append({
                    "parent_id": node_id,
                    "child_id": node_ids[c_idx],
                    "p_thick": p_thick,
                    "c_thick": c_thick,
                    "jump": jump,
                    "p_desc": descendant_counts[i],
                    "c_desc": descendant_counts[c_idx]
                })

    junction_jumps.sort(key=lambda x: x["jump"], reverse=True)
    for j in junction_jumps[:5]:
        print(f"Junction ID:{j['parent_id']} -> ID:{j['child_id']} | Jump:{j['jump']:.2f} ({j['p_thick']:.2f} -> {j['c_thick']:.2f}) | Desc:{j['p_desc']} -> {j['c_desc']}")

if __name__ == "__main__":
    analyze_variant("var_0004")
    analyze_variant("var_0007")
