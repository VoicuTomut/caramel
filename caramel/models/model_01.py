"""
This is a dummy PyTorch Geometric model that takes a graph as input and outputs the moment of contraction for each node.
One of this  project purposes is to improve this to one more functional model.
"""
# https://github.com/deepfindr/gnn-project/blob/main/model.py
# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html

import torch
# import torch.nn.functional as F

from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv


# GINConv
# PointNetConv
class Model01(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(Model01, self).__init__()

        embedding_size = model_params["embedding_size"]  # 200
        n_heads = model_params["n_heads"]  # 4
        dropout_rate = model_params["dropout_rate"]  # 0.9
        edge_dim = model_params["edge_dim"]  # 2
        self.n_layers = model_params["model_layers"]

        # Transformation layer
        self.conv1 = TransformerConv(feature_size,
                                     embedding_size,
                                     heads=n_heads,
                                     dropout=dropout_rate,
                                     edge_dim=edge_dim,
                                     beta=True)
        self.transf1 = Linear(embedding_size * n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        # middle
        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        for i in range(self.n_layers):
            self.conv_layers.append(TransformerConv(embedding_size,
                                                    embedding_size,
                                                    heads=n_heads,
                                                    dropout=dropout_rate,
                                                    edge_dim=edge_dim,
                                                    beta=True))
            self.transf_layers.append(Linear(embedding_size * n_heads, embedding_size))

        self.conv2 = TransformerConv(embedding_size,
                                     1,
                                     heads=n_heads,
                                     dropout=dropout_rate,
                                     edge_dim=edge_dim,
                                     beta=True)
        self.transf2 = Linear(1 * n_heads, 1)
        self.bn2 = BatchNorm1d(1)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.transf1(x)
        x = self.bn1(x)

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            # print("i",i)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.transf2(x)
        x = self.bn2(x)

        return x
