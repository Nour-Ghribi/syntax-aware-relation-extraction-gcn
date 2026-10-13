"""
Basic operations on trees.
"""

import numpy as np

class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self,child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size', None):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth', None):
            return self._depth
        count = 0
        if self.num_children>0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

def head_to_tree(head, tokens, len_, prune, subj_pos, obj_pos):
    """
    Convert a sequence of head indexes into a tree object.
    """
    tokens = tokens[:len_].tolist()
    head = head[:len_].tolist()
    root = None

    if prune < 0:
        nodes = [Tree() for _ in head]

        for i in range(len(nodes)):
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = -1 # just a filler
            if h == 0:
                root = nodes[i]
            else:
                nodes[h-1].add_child(nodes[i])
    else:
        path_nodes, lca, dist = pruning_info(head, len_, subj_pos, obj_pos)

        highest_node = lca
        nodes = [Tree() if dist[i] <= prune else None for i in range(len_)]

        for i in range(len(nodes)):
            if nodes[i] is None:
                continue
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = dist[i]
            if h > 0 and i != highest_node:
                assert nodes[h-1] is not None
                nodes[h-1].add_child(nodes[i])

        root = nodes[highest_node]

    assert root is not None
    return root

def pruning_info(head, len_, subj_pos, obj_pos):
    """
    Return path-centric pruning metadata for a dependency tree.

    ``subj_pos`` and ``obj_pos`` use the repository convention: entity tokens
    are marked with 0, surrounding tokens with relative distances.
    """
    if not isinstance(head, list):
        head = head[:len_].tolist()
    else:
        head = head[:len_]

    subj_indices = [i for i in range(len_) if subj_pos[i] == 0]
    obj_indices = [i for i in range(len_) if obj_pos[i] == 0]
    if len(subj_indices) == 0 or len(obj_indices) == 0:
        raise ValueError("Subject and object spans must each contain at least one token.")

    cas = None
    subj_ancestors = set(subj_indices)
    for s in subj_indices:
        h = head[s]
        tmp = [s]
        while h > 0:
            tmp += [h - 1]
            subj_ancestors.add(h - 1)
            h = head[h - 1]
        cas = set(tmp) if cas is None else cas.intersection(tmp)

    obj_ancestors = set(obj_indices)
    for o in obj_indices:
        h = head[o]
        tmp = [o]
        while h > 0:
            tmp += [h - 1]
            obj_ancestors.add(h - 1)
            h = head[h - 1]
        cas = set(tmp) if cas is None else cas.intersection(tmp)

    if len(cas) == 0:
        raise ValueError("Could not find a common ancestor for subject and object spans.")

    if len(cas) == 1:
        lca = list(cas)[0]
    else:
        child_count = {k: 0 for k in cas}
        for ca in cas:
            if head[ca] > 0 and head[ca] - 1 in cas:
                child_count[head[ca] - 1] += 1
        lca = next(ca for ca in cas if child_count[ca] == 0)

    path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)
    path_nodes.add(lca)
    dist = [-1 if i not in path_nodes else 0 for i in range(len_)]

    for i in range(len_):
        if dist[i] < 0:
            stack = [i]
            while stack[-1] >= 0 and stack[-1] not in path_nodes:
                stack.append(head[stack[-1]] - 1)

            if stack[-1] in path_nodes:
                for d, j in enumerate(reversed(stack)):
                    dist[j] = d
            else:
                for j in stack:
                    if j >= 0 and dist[j] < 0:
                        dist[j] = int(1e4)

    return path_nodes, lca, dist

def tree_to_adj(sent_len, tree, directed=True, self_loop=False):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret

def tree_to_dist(sent_len, tree):
    ret = -1 * np.ones(sent_len, dtype=np.int64)

    for node in tree:
        ret[node.idx] = node.dist

    return ret
