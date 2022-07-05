"""

"""

import torch
import torch.nn.functional as F

from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv



class Dummy_Net(torch.nn.Module):
    def __init__(self,feature_size):
        super(Dummy_Net, self).__init__()

        embedding_size = 128
        n_heads = 4
        dropout_rate = 0.9
        edge_dim = 1
        # Transformation layer
        self.conv1 = TransformerConv(feature_size,
                                     embedding_size,
                                     heads=n_heads,
                                     dropout=dropout_rate,
                                     edge_dim=edge_dim,
                                     beta=True)
        self.transf1 = Linear(embedding_size * n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)


    def forward(self, x, edge_index, edge_attr):
            print("x0:", x.shape)
            print("edge_index:", edge_index.shape)
            print("edge_attr:", edge_attr.shape)
            x = self.conv1(x, edge_index, edge_attr)
            x = self.transf1(x)
            x = self.bn1(x)
            print("x.shape:",x.shape[0])
            print("x1:",x)


            return x