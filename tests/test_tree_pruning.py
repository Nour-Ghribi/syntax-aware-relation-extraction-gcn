import numpy as np

from loader import get_positions
from tree import head_to_tree, pruning_info, tree_to_adj, tree_to_dist


def toy_dependency_example():
    tokens = np.arange(5)
    # John founded Acme in Paris
    # 1-based heads: John -> founded, founded -> ROOT, Acme -> founded,
    # in -> Paris, Paris -> founded.
    heads = np.asarray([2, 0, 2, 5, 2])
    subj_pos = get_positions(0, 0, len(tokens))
    obj_pos = get_positions(2, 2, len(tokens))
    return tokens, heads, subj_pos, obj_pos


def test_dependency_tree_construction_without_pruning():
    tokens, heads, subj_pos, obj_pos = toy_dependency_example()
    tree = head_to_tree(heads, tokens, len(tokens), -1, subj_pos, obj_pos)
    adj = tree_to_adj(len(tokens), tree, directed=True)

    assert tree.idx == 1
    assert adj[1, 0] == 1
    assert adj[1, 2] == 1
    assert adj[1, 4] == 1
    assert adj[4, 3] == 1


def test_shortest_dependency_path_metadata():
    tokens, heads, subj_pos, obj_pos = toy_dependency_example()
    path_nodes, lca, distances = pruning_info(heads, len(tokens), subj_pos, obj_pos)

    assert lca == 1
    assert path_nodes == {0, 1, 2}
    assert distances == [0, 0, 0, 2, 1]


def test_pruning_k_zero_keeps_only_subject_object_path():
    tokens, heads, subj_pos, obj_pos = toy_dependency_example()
    tree = head_to_tree(heads, tokens, len(tokens), 0, subj_pos, obj_pos)

    assert set(node.idx for node in tree) == {0, 1, 2}
    assert tree_to_dist(len(tokens), tree).tolist() == [0, 0, 0, -1, -1]


def test_pruning_k_one_retains_one_hop_context():
    tokens, heads, subj_pos, obj_pos = toy_dependency_example()
    tree = head_to_tree(heads, tokens, len(tokens), 1, subj_pos, obj_pos)

    assert set(node.idx for node in tree) == {0, 1, 2, 4}
    assert tree_to_dist(len(tokens), tree).tolist() == [0, 0, 0, -1, 1]


def test_adjacency_matrix_can_be_undirected_with_self_loops():
    tokens, heads, subj_pos, obj_pos = toy_dependency_example()
    tree = head_to_tree(heads, tokens, len(tokens), 1, subj_pos, obj_pos)
    adj = tree_to_adj(len(tokens), tree, directed=False, self_loop=True)

    assert adj.shape == (5, 5)
    assert adj[1, 4] == 1
    assert adj[4, 1] == 1
    assert adj[4, 4] == 1
    assert adj[3].sum() == 0
