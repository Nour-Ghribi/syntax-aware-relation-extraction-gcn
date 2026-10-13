import json

import matplotlib

matplotlib.use("Agg")

from tools.inspect_architectures import inspect_architectures, write_reports
from tools.visualize_dependency_pruning import load_example, pruning_summary, visualize, visualize_comparison


def test_visualization_file_generation(tmp_path):
    example_path = tmp_path / "sample.json"
    output_path = tmp_path / "pruning.svg"
    example = {
        "id": "toy",
        "relation": "org:founded_by",
        "token": ["John", "founded", "Acme", "in", "Paris"],
        "subj_start": 0,
        "subj_end": 0,
        "obj_start": 2,
        "obj_end": 2,
        "subj_type": "PERSON",
        "obj_type": "ORGANIZATION",
        "stanford_pos": ["NNP", "VBD", "NNP", "IN", "NNP"],
        "stanford_ner": ["PERSON", "O", "ORGANIZATION", "O", "LOCATION"],
        "stanford_head": ["2", "0", "2", "5", "2"],
        "stanford_deprel": ["nsubj", "ROOT", "dobj", "case", "nmod"],
    }
    example_path.write_text(json.dumps([example]), encoding="utf8")

    loaded = load_example(example_path, 0)
    visualize(loaded, 1, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_pruning_comparison_generation_and_summary(tmp_path):
    output_path = tmp_path / "comparison.svg"
    example = {
        "id": "toy",
        "relation": "org:founded_by",
        "token": ["John", "founded", "Acme", "in", "Paris"],
        "subj_start": 0,
        "subj_end": 0,
        "obj_start": 2,
        "obj_end": 2,
        "subj_type": "PERSON",
        "obj_type": "ORGANIZATION",
        "stanford_pos": ["NNP", "VBD", "NNP", "IN", "NNP"],
        "stanford_ner": ["PERSON", "O", "ORGANIZATION", "O", "LOCATION"],
        "stanford_head": ["2", "0", "2", "5", "2"],
        "stanford_deprel": ["nsubj", "ROOT", "dobj", "case", "nmod"],
    }

    generated_path, summary = visualize_comparison(example, output_path)
    counts = [row["retained_nodes"] for row in pruning_summary(example)["rows"]]

    assert generated_path.exists()
    assert output_path.stat().st_size > 0
    assert counts == [3, 4, 5]
    assert summary["subject"] == "John"
    assert summary["object"] == "Acme"


def test_architecture_inspection_reports_cpu_forward_passes(tmp_path):
    rows = inspect_architectures()
    markdown_path = tmp_path / "architecture.md"
    json_path = tmp_path / "architecture.json"
    write_reports(rows, markdown_path, json_path)

    assert [row["architecture"] for row in rows] == ["GCN", "C-GCN"]
    assert all(row["cpu_forward_pass"] for row in rows)
    assert all(row["logits_shape"] == [2, 5] for row in rows)
    assert markdown_path.exists()
    assert json_path.exists()
