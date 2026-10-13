"""Inspect GCN and C-GCN model structure with synthetic CPU inputs.

This is a structural check, not a training or predictive-performance benchmark.
It instantiates the repository models, runs one CPU forward pass with synthetic
tensors, and writes lightweight Markdown/JSON reports.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import constant
from gcn import GCNClassifier


def base_options(rnn):
    return {
        "vocab_size": 64,
        "emb_dim": 16,
        "pos_dim": 8,
        "ner_dim": 8,
        "hidden_dim": 12,
        "num_layers": 2,
        "input_dropout": 0.0,
        "gcn_dropout": 0.0,
        "word_dropout": 0.0,
        "rnn": rnn,
        "rnn_hidden": 6,
        "rnn_layers": 1,
        "rnn_dropout": 0.0,
        "mlp_layers": 1,
        "num_class": 5,
        "topn": 64,
        "cuda": False,
        "prune_k": -1,
        "pooling": "max",
    }


def synthetic_inputs(batch_size=2, seq_len=5):
    words = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 0]], dtype=torch.long)
    masks = words.eq(constant.PAD_ID)
    pos = torch.ones(batch_size, seq_len, dtype=torch.long) * 2
    ner = torch.ones(batch_size, seq_len, dtype=torch.long) * 2
    deprel = torch.ones(batch_size, seq_len, dtype=torch.long) * 2
    head = torch.tensor([[2, 0, 2, 3, 4], [2, 0, 2, 3, 0]], dtype=torch.long)
    subj_pos = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long)
    obj_pos = torch.tensor([[-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]], dtype=torch.long)
    subj_type = torch.ones(batch_size, 1, dtype=torch.long) * 2
    obj_type = torch.ones(batch_size, 1, dtype=torch.long) * 2
    return (words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def inspect_variant(name, options):
    torch.manual_seed(1234)
    with contextlib.redirect_stdout(io.StringIO()):
        model = GCNClassifier(options)
    model.eval()
    report = {
        "architecture": name,
        "bilstm_enabled": bool(options["rnn"]),
        "gcn_layers": options["num_layers"],
        "hidden_dim": options["hidden_dim"],
        "pooling": options["pooling"],
        "parameters": parameter_count(model),
        "cpu_forward_pass": False,
        "logits_shape": None,
        "pooled_shape": None,
    }
    with torch.no_grad():
        logits, pooled = model(synthetic_inputs())
    report["cpu_forward_pass"] = True
    report["logits_shape"] = list(logits.shape)
    report["pooled_shape"] = list(pooled.shape)
    return report


def inspect_architectures():
    return [
        inspect_variant("GCN", base_options(rnn=False)),
        inspect_variant("C-GCN", base_options(rnn=True)),
    ]


def markdown_table(rows):
    header = [
        "Architecture",
        "BiLSTM",
        "GCN layers",
        "Hidden dim",
        "Pooling",
        "Parameters",
        "CPU forward",
        "Logits shape",
        "Pooled shape",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append(
            "| {architecture} | {bilstm_enabled} | {gcn_layers} | {hidden_dim} | {pooling} | {parameters} | {cpu_forward_pass} | {logits_shape} | {pooled_shape} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def write_reports(rows, markdown_path, json_path):
    markdown_path = Path(markdown_path)
    json_path = Path(json_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_table(rows), encoding="utf8")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf8")


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect GCN/C-GCN architecture structure.")
    parser.add_argument("--markdown", default=str(ROOT / "reports" / "architecture_comparison.md"))
    parser.add_argument("--json", default=str(ROOT / "reports" / "architecture_comparison.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    rows = inspect_architectures()
    write_reports(rows, args.markdown, args.json)
    print(markdown_table(rows))
    print("Saved Markdown report to {}".format(args.markdown))
    print("Saved JSON report to {}".format(args.json))


if __name__ == "__main__":
    main()
