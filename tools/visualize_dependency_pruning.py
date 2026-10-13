"""Visualize path-centric dependency tree pruning on a TACRED-style example.

This tool does not load trained weights and does not require the full TACRED
dataset. It works with the small sample JSON files included in this repository.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loader import get_positions
from tree import head_to_tree, pruning_info, tree_to_adj


def load_example(path, index):
    with open(path, encoding="utf8") as infile:
        data = json.load(infile)
    if index < 0 or index >= len(data):
        raise IndexError("Example index {} is outside 0..{}.".format(index, len(data) - 1))
    return data[index]


def graph_from_heads(example):
    graph = nx.DiGraph()
    tokens = example["token"]
    heads = [int(x) for x in example["stanford_head"]]
    deprels = example.get("stanford_deprel", ["dep"] * len(tokens))

    for i, token in enumerate(tokens):
        graph.add_node(i, label="{}:{}".format(i, token))
    for i, head in enumerate(heads):
        if head > 0:
            graph.add_edge(head - 1, i, label=deprels[i])
    return graph


def node_groups(example, k):
    length = len(example["token"])
    heads = [int(x) for x in example["stanford_head"]]
    subj_pos = get_positions(example["subj_start"], example["subj_end"], length)
    obj_pos = get_positions(example["obj_start"], example["obj_end"], length)
    path_nodes, _lca, distances = pruning_info(heads, length, subj_pos, obj_pos)
    retained = {i for i, distance in enumerate(distances) if distance <= k}
    subject = set(range(example["subj_start"], example["subj_end"] + 1))
    object_ = set(range(example["obj_start"], example["obj_end"] + 1))
    return subject, object_, path_nodes, retained, distances


def pruning_summary(example, ks=(0, 1, 2)):
    """Return retained-node statistics for the requested pruning distances."""
    length = len(example["token"])
    subject, object_, path_nodes, _retained, distances = node_groups(example, max(ks))
    rows = []
    for k in ks:
        retained = {i for i, distance in enumerate(distances) if distance <= k}
        rows.append(
            {
                "k": k,
                "retained_nodes": len(retained),
                "total_nodes": length,
                "retained_percent": round(100.0 * len(retained) / length, 2),
                "retained_tokens": [example["token"][i] for i in sorted(retained)],
            }
        )
    return {
        "example_id": example.get("id", "unknown"),
        "relation": example.get("relation", "unknown"),
        "subject": " ".join(example["token"][example["subj_start"] : example["subj_end"] + 1]),
        "object": " ".join(example["token"][example["obj_start"] : example["obj_end"] + 1]),
        "shortest_path_tokens": [example["token"][i] for i in sorted(path_nodes)],
        "rows": rows,
        "subject_nodes": sorted(subject),
        "object_nodes": sorted(object_),
    }


def colors_for_nodes(nodes, subject, object_, path_nodes, retained):
    colors = []
    for node in nodes:
        if node in subject:
            colors.append("#e45756")
        elif node in object_:
            colors.append("#4c78a8")
        elif node in path_nodes:
            colors.append("#f2b134")
        elif node in retained:
            colors.append("#54a24b")
        else:
            colors.append("#d6d6d6")
    return colors


def draw_graph(ax, graph, title, subject, object_, path_nodes, retained):
    undirected = graph.to_undirected()
    pos = nx.spring_layout(undirected, seed=7, k=1.2)
    labels = nx.get_node_attributes(graph, "label")
    edge_labels = nx.get_edge_attributes(graph, "label")
    nodes = list(graph.nodes())

    nx.draw_networkx_edges(graph, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=12, width=1.3)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=colors_for_nodes(nodes, subject, object_, path_nodes, retained),
        node_size=900,
        edgecolors="#2b2b2b",
        linewidths=0.7,
    )
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=7)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_size=6)
    ax.set_title(title)
    ax.axis("off")


def pruned_graph_from_example(example, k):
    tokens = np.arange(len(example["token"]))
    subj_pos = get_positions(example["subj_start"], example["subj_end"], len(tokens))
    obj_pos = get_positions(example["obj_start"], example["obj_end"], len(tokens))
    subject, object_, path_nodes, retained, _distances = node_groups(example, k)
    pruned_tree = head_to_tree(
        np.asarray([int(x) for x in example["stanford_head"]]),
        tokens,
        len(tokens),
        k,
        subj_pos,
        obj_pos,
    )
    pruned_adj = tree_to_adj(len(tokens), pruned_tree, directed=True, self_loop=False)
    pruned_graph = nx.DiGraph()
    for node in sorted(retained):
        pruned_graph.add_node(node, label="{}:{}".format(node, example["token"][node]))
    for source, target in zip(*np.where(pruned_adj > 0)):
        pruned_graph.add_edge(source, target, label=example["stanford_deprel"][target])
    return pruned_graph, subject, object_, path_nodes, retained


def visualize(example, k, output_path):
    graph = graph_from_heads(example)
    pruned_graph, subject, object_, path_nodes, retained = pruned_graph_from_example(example, k)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    draw_graph(axes[0], graph, "Original dependency graph", subject, object_, path_nodes, retained)
    draw_graph(axes[1], pruned_graph, "Pruned graph (k={})".format(k), subject, object_, path_nodes, retained)
    fig.suptitle(
        "Path-centric pruning: subject={}, object={}, relation={}".format(
            example["subj_type"], example["obj_type"], example["relation"]
        ),
        fontsize=12,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def visualize_comparison(example, output_path):
    """Create a four-panel comparison of the full tree and k=0/1/2 pruning."""
    full_graph = graph_from_heads(example)
    subject, object_, path_nodes, _retained, _distances = node_groups(example, 2)
    panels = [("Full dependency tree", full_graph, set(full_graph.nodes()))]
    for k in (0, 1, 2):
        graph, _subject, _object, _path, retained = pruned_graph_from_example(example, k)
        panels.append(("Pruned tree (k={})".format(k), graph, retained))

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), constrained_layout=True)
    total_nodes = len(example["token"])
    for ax, (title, graph, retained) in zip(axes.ravel(), panels):
        retained_pct = 100.0 * len(retained) / total_nodes
        draw_graph(
            ax,
            graph,
            "{}\n{} / {} nodes retained ({:.1f}%)".format(title, len(retained), total_nodes, retained_pct),
            subject,
            object_,
            path_nodes,
            retained,
        )

    summary = pruning_summary(example)
    fig.suptitle(
        "Dependency pruning comparison for example {id}: subject='{subj}', object='{obj}', relation='{rel}'".format(
            id=summary["example_id"],
            subj=summary["subject"],
            obj=summary["object"],
            rel=summary["relation"],
        ),
        fontsize=13,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize dependency path pruning.")
    parser.add_argument("--input", default=str(ROOT / "train.json"), help="TACRED-style JSON file.")
    parser.add_argument("--index", type=int, default=0, help="Example index in the JSON file.")
    parser.add_argument("--k", type=int, default=1, help="Keep nodes within distance k of the subject-object dependency path.")
    parser.add_argument("--output", default=str(ROOT / "assets" / "dependency_pruning_k1.svg"), help="SVG/PNG output path.")
    parser.add_argument("--comparison", action="store_true", help="Generate a full/k=0/k=1/k=2 comparison figure.")
    return parser.parse_args()


def main():
    args = parse_args()
    example = load_example(args.input, args.index)
    if args.comparison:
        output_path, summary = visualize_comparison(example, args.output)
        print("Saved dependency pruning comparison to {}".format(output_path))
        for row in summary["rows"]:
            print(
                "k={k}: {retained_nodes}/{total_nodes} nodes retained ({retained_percent}%)".format(
                    **row
                )
            )
    else:
        output_path = visualize(example, args.k, args.output)
        print("Saved dependency pruning visualization to {}".format(output_path))


if __name__ == "__main__":
    main()
