# src: https://github.com/qipeng/gcn-over-pruned-trees/blob/master/model/tree.py
"""
Basic operations on trees.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from third_party.processors.constants import DEP_TAG_MAP


class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """

    def __init__(self):
        self.parent = None
        self.idx = None
        self.token = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def print(self, level):
        for i in range(1, level):
            print('|----', end='')
        print(self.token)
        for i in range(self.num_children):
            self.children[i].print(level + 1)

    def size(self):
        if getattr(self, '_size', False):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def height(self):
        if getattr(self, '_height', False):
            return self._height
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_height = self.children[i].height()
                if child_height > count:
                    count = child_height
            count += 1
        self._height = count
        return self._height

    def depth(self):
        if getattr(self, '_depth', False):
            return self._depth
        count = 0
        if self.parent:
            count += self.parent.depth()
            count += 1
        self._depth = count
        return self._depth

    def delete(self):
        for i in range(self.num_children):
            self.parent.add_child(self.children[i])
            self.children[i].parent = self.parent
        index = None
        for i in range(self.parent.num_children):
            if self.parent.children[i].idx == self.idx:
                index = i
                break
        self.parent.children.pop(index)
        self.parent.num_children -= 1

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x


def tree_to_adj(sent_len, tree, directed=False, self_loop=False):
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


def head_to_tree(head, tokens=None):
    """
    Convert a sequence of head indexes into a tree object.
    """
    root = None
    nodes = [Tree() for _ in head]
    for i in range(len(nodes)):
        h = head[i]
        nodes[i].idx = i
        if tokens is not None:
            nodes[i].token = tokens[i]
        if h == 0:
            root = nodes[i]
        else:
            nodes[h - 1].add_child(nodes[i])

    assert root is not None
    return root, nodes


def heads_to_dist_mat(head, tokens, directed=False):
    root, _ = head_to_tree(head, tokens)
    adj_mat = tree_to_adj(root.size(), root, directed=directed, self_loop=False)
    dist_matrix = shortest_path(csgraph=csr_matrix(adj_mat), directed=directed)
    return dist_matrix


def root_to_dist_mat(root, directed=False):
    adj_mat = tree_to_adj(root.size(), root, directed=directed, self_loop=False)
    dist_matrix = shortest_path(csgraph=csr_matrix(adj_mat), directed=directed)
    return dist_matrix


def adj_mat_to_dist_mat(adj_mat, directed=False):
    dist_matrix = shortest_path(csgraph=csr_matrix(adj_mat), directed=directed)
    return dist_matrix


def dist_to_root(head, tokens):
    root, nodes = head_to_tree(head, tokens)
    distances = []
    for i in range(len(nodes)):
        distances.append(nodes[i].depth())

    return distances


def ancestorMatrixRec(root, anc, mat):
    # base case  
    if root == None:
        return 0

    # Update all ancestors of current node  
    data = root.idx
    for i in range(len(anc)):
        mat[anc[i]][data] = 1

    # Push data to list of ancestors  
    anc.append(data)

    # Traverse all the subtrees 
    for c in root.children:
        ancestorMatrixRec(c, anc, mat)

    # Remove data from list the list of ancestors  
    # as all descendants of it are processed now.  
    anc.pop(-1)


def heads_to_ancestor_matrix(heads, tokens):
    mat = np.zeros((len(tokens), len(tokens)), dtype=np.int32)
    root, _ = head_to_tree(heads, tokens)
    ancestorMatrixRec(root, [], mat)
    np.fill_diagonal(mat, 1)
    return mat


def dep_path_matrix(heads, tokens, dep_labels, root=None):
    assert len(heads) == len(tokens) == len(dep_labels)
    if root is None:
        root, _ = head_to_tree(heads, tokens)

    def find_path(i, j, preds):
        if predecessors[i, j] == i:
            return preds
        preds.append(predecessors[i, j])
        find_path(i, predecessors[i, j], preds)

    adj_mat = tree_to_adj(root.size(), root, directed=False, self_loop=False)
    _, predecessors = shortest_path(
        csgraph=csr_matrix(adj_mat), directed=False, return_predecessors=True
    )

    max_path_length = 0
    token_to_token_paths = {}
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            preds = []
            find_path(i, j, preds)
            token_ids = [i] + preds[::-1] + [j]
            if len(token_ids) - 1 > max_path_length:
                max_path_length = len(token_ids) - 1

            path_labels = []
            for k in range(len(token_ids) - 1):
                # token[k+1] is the head of token[k]
                if heads[token_ids[k]] - 1 == token_ids[k + 1]:
                    path_labels.append(dep_labels[token_ids[k]])
                # token[k] is the head of token[k+1]
                elif heads[token_ids[k + 1]] - 1 == token_ids[k]:
                    path_labels.append(dep_labels[token_ids[k + 1]])
                else:
                    raise ValueError()
            token_to_token_paths['{}.{}'.format(i, j)] = path_labels

    path_matrix = np.empty((len(tokens), len(tokens), max_path_length), dtype=np.int32)
    for i in range(len(tokens)):
        for j in range(i, len(tokens)):
            if i == j:
                labels = ['<self>'] + ['<pad>'] * (max_path_length - 1)
                path_matrix[i, j] = [DEP_TAG_MAP[l] for l in labels]
            else:
                labels = token_to_token_paths['{}.{}'.format(i, j)]
                pad_length = max_path_length - len(labels)
                labels = labels + ['<pad>'] * pad_length
                labels = [DEP_TAG_MAP[l] for l in labels]
                path_matrix[i, j] = labels
                path_matrix[j, i] = labels

    return path_matrix


def get_dep_path(heads, tokens, dep_labels, root=None):
    assert len(heads) == len(tokens) == len(dep_labels)
    if root is None:
        root, _ = head_to_tree(heads, tokens)

    def find_path(i, j, preds):
        if predecessors[i, j] == i:
            return preds
        preds.append(predecessors[i, j])
        find_path(i, predecessors[i, j], preds)

    adj_mat = tree_to_adj(root.size(), root, directed=False, self_loop=False)
    _, predecessors = shortest_path(
        csgraph=csr_matrix(adj_mat), directed=False, return_predecessors=True
    )

    max_path_length = 0
    token_to_token_paths = {}
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            preds = []
            find_path(i, j, preds)
            token_ids = [i] + preds[::-1] + [j]
            if len(token_ids) - 1 > max_path_length:
                max_path_length = len(token_ids) - 1

            path_labels = []
            for k in range(len(token_ids) - 1):
                # token[k+1] is the head of token[k]
                if heads[token_ids[k]] - 1 == token_ids[k + 1]:
                    path_labels.append(dep_labels[token_ids[k]])
                # token[k] is the head of token[k+1]
                elif heads[token_ids[k + 1]] - 1 == token_ids[k]:
                    path_labels.append(dep_labels[token_ids[k + 1]])
                else:
                    raise ValueError()
            token_to_token_paths['{}.{}'.format(i, j)] = path_labels

    return token_to_token_paths


def get_path_matrix(heads, tokens, dep_labels, token_to_token_paths):
    assert len(heads) == len(tokens) == len(dep_labels)

    max_path_length = max([len(v) for k, v in token_to_token_paths.items()])
    path_matrix = np.empty((len(tokens), len(tokens), max_path_length), dtype=np.int32)
    for i in range(len(tokens)):
        for j in range(i, len(tokens)):
            if i == j:
                labels = ['<self>'] + ['<pad>'] * (max_path_length - 1)
                path_matrix[i, j] = [DEP_TAG_MAP[l] for l in labels]
            else:
                labels = token_to_token_paths['{}.{}'.format(i, j)]
                pad_length = max_path_length - len(labels)
                labels = labels + ['<pad>'] * pad_length
                labels = [DEP_TAG_MAP[l] for l in labels]
                path_matrix[i, j] = labels
                path_matrix[j, i] = labels

    return path_matrix


if __name__ == '__main__':
    tokens = ['The', 'increase', 'reflects', 'lower', 'credit', 'losses']
    heads = [2, 3, 0, 6, 6, 3]
    anc_mat = heads_to_ancestor_matrix(heads, tokens)
    # print(anc_mat)
    # [[1 1 1 0 0 0]
    #  [0 1 1 0 0 0]
    #  [0 0 1 0 0 0]
    #  [0 0 1 1 0 1]
    #  [0 0 1 0 1 1]
    #  [0 0 1 0 0 1]]
