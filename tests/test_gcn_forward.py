import torch

import constant
from gcn import GCNClassifier


def make_options(rnn=False):
    return {
        "vocab_size": 20,
        "emb_dim": 8,
        "pos_dim": 4,
        "ner_dim": 4,
        "hidden_dim": 6,
        "num_layers": 2,
        "input_dropout": 0.0,
        "gcn_dropout": 0.0,
        "rnn": rnn,
        "rnn_hidden": 3,
        "rnn_layers": 1,
        "rnn_dropout": 0.0,
        "mlp_layers": 1,
        "num_class": 3,
        "topn": 20,
        "cuda": False,
        "prune_k": -1,
        "pooling": "max",
    }


def make_inputs():
    batch_size = 2
    seq_len = 4
    words = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 0]])
    masks = words.eq(constant.PAD_ID)
    pos = torch.ones(batch_size, seq_len, dtype=torch.long) * 2
    ner = torch.ones(batch_size, seq_len, dtype=torch.long) * 2
    deprel = torch.ones(batch_size, seq_len, dtype=torch.long) * 2
    head = torch.tensor([[2, 0, 2, 3], [2, 0, 2, 0]])
    subj_pos = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    obj_pos = torch.tensor([[-2, -1, 0, 1], [-2, -1, 0, 1]])
    subj_type = torch.ones(batch_size, 1, dtype=torch.long) * 2
    obj_type = torch.ones(batch_size, 1, dtype=torch.long) * 2
    return words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type


def test_gcn_forward_pass_shape_with_synthetic_tensors():
    opt = make_options(rnn=False)
    model = GCNClassifier(opt)
    batch_size = 2

    logits, pooled = model(make_inputs())

    assert logits.shape == (batch_size, opt["num_class"])
    assert pooled.shape == (batch_size, opt["hidden_dim"])
    assert not next(model.parameters()).is_cuda


def test_cgcn_forward_pass_shape_with_synthetic_tensors_on_cpu():
    opt = make_options(rnn=True)
    model = GCNClassifier(opt)

    logits, pooled = model(make_inputs())

    assert logits.shape == (2, opt["num_class"])
    assert pooled.shape == (2, opt["hidden_dim"])
    assert not next(model.parameters()).is_cuda
