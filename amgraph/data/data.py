import os
from collections import defaultdict
import numpy as np
import torch
from torch_geometric.typing import Adj
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from sklearn.model_selection import train_test_split


def is_binary(data):
    return data in ['cora', 'citeseer', 'computers', 'photo', 'cs']


def is_continuous(data):
    return data in ['pubmed', 'coauthor', 'cs', 'arxiv']


def validate_edges(edges):
    """
    Validate the edges of a graph with various criteria.
    """
    # No self-loops
    for src, dst in edges.t():
        if src.item() == dst.item():
            raise ValueError(f"{src} has self-loops")

    # Each edge (a, b) appears only once.
    m = defaultdict(lambda: set())
    for src, dst in edges.t():
        src = src.item()
        dst = dst.item()
        if dst in m[src]:
            raise ValueError(f"({src}->{dst}) appears more than once")
        m[src].add(dst)

    # Each pair (a, b) and (b, a) exists together.
    for src, neighbors in m.items():
        for dst in neighbors:
            if src not in m[dst]:
                raise ValueError(f"({src}->{dst}) lack of inverse edge")


class Dataset:

    def __init__(self, data_name: str, edges: Adj, x: torch.Tensor, y: torch.Tensor, trn_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor, num_classes: int):
        self.data_name = data_name
        self.edges = edges
        self.x = x
        self.y = y
        self.trn_mask = trn_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_classes = num_classes
        self.num_nodes = x.size(0)
        self.num_attrs = x.size(1) if len(x.size()) > 1 else 1
        self.is_binary = is_binary(data_name)
        self.is_continuous = is_continuous(data_name)

    def to(self, device):
        self.edges = self.edges.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.trn_mask = self.trn_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        return self


def load_data(data_name, split=None, seed=None, verbose=False) -> Dataset:
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    if data_name == 'arxiv':
        from ogb.nodeproppred import PygNodePropPredDataset
        data = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
        data.data.edge_index = to_undirected(data.data.edge_index)
    elif data_name == 'cora':
        data = Planetoid(root, 'Cora')
    elif data_name == 'citeseer':
        data = Planetoid(root, 'CiteSeer')
    elif data_name == 'pubmed':
        data = Planetoid(root, 'PubMed')
    elif data_name == 'computers':
        data = Amazon(root, 'Computers')
    elif data_name == 'photo':
        data = Amazon(root, 'Photo')
    elif data_name == 'cs' or data_name == 'coauthor':
        data = Coauthor(root, 'CS')
    elif data_name == 'physics':
        data = Coauthor(root, 'Physics')
    else:
        raise ValueError(data_name)

    dat = data.data

    node_x = dat.x
    node_y = dat.y.squeeze()
    edges = dat.edge_index

    validate_edges(edges)

    if split is None:
        if hasattr(dat, 'train_mask'):
            trn_mask = dat.train_mask
            val_mask = dat.val_mask
            trn_nodes = torch.nonzero(trn_mask).view(-1)
            val_nodes = torch.nonzero(val_mask).view(-1)
            test_nodes = torch.nonzero(~(trn_mask | val_mask)).view(-1)
        else:
            trn_nodes, val_nodes, test_nodes = None, None, None
    elif len(split) == 3 and sum(split) == 1:
        trn_size, val_size, test_size = split
        indices = np.arange(node_x.shape[0])
        trn_nodes, test_nodes = train_test_split(indices, test_size=test_size, random_state=seed,
                                                 stratify=node_y)
        trn_nodes, val_nodes = train_test_split(trn_nodes, test_size=val_size / (trn_size + val_size),
                                                random_state=seed, stratify=node_y[trn_nodes])

        trn_nodes = torch.from_numpy(trn_nodes).to(torch.long)
        val_nodes = torch.from_numpy(val_nodes).to(torch.long)
        test_nodes = torch.from_numpy(test_nodes).to(torch.long)
    else:
        raise ValueError(split)

    if verbose:
        print('Data:', data_name)
        print('Number of nodes:', node_x.size(0))
        print('Number of edges:', edges.size(1) // 2)
        print('Number of features:', node_x.size(1))
        print('Ratio of nonzero features:', (node_x > 0).float().mean().item())
        print('Number of classes:', node_y.max().item() + 1 if node_y is not None else 0)
        print()

    # return data
    return Dataset(data_name=data_name, edges=edges, x=node_x, y=node_y, trn_mask=trn_nodes, val_mask=val_nodes,
                   test_mask=test_nodes, num_classes=data.num_classes)


if __name__ == '__main__':
    load_data('arxiv')
    print('Data process done!')
