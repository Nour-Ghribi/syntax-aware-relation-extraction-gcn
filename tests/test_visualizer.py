import json

import matplotlib

matplotlib.use("Agg")

from tools.visualize_dependency_pruning import load_example, visualize


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
