"""

"""

import torch
import torch.nn.functional as F

from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv

embedding_size = 128

class Dummy_Net(torch.nn.Module):
    def __init__(self,feature_size):
        super(Dummy_Net, self).__init__()


        n_heads = model_parameters["n_heads"]
        dropout_rate = model_parameters["dropout_rate"]
        edge_dim = model_parameters["edge_dim"]
        # Transformation layer
        self.conv1 = TransformerConv(feature_size,
                                     embedding_size,
                                     heads=n_heads,
                                     dropout=dropout_rate,
                                     edge_dim=edge_dim,
                                     beta=True)
        self.transf1 = Linear(embedding_size * n_heads, embedding_size)

    def forward(self, x, edge_attr, edge_index, batch_index):
            print("x0:", x)
            x = self.conv1(x, edge_index, edge_attr)
            x = torch.relu(self.transf1(x))
            x = self.bn1(x)
            print("x1:",x)
            return x