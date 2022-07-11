"""
Example of custom layer in pytorch.
"""

import torch
import torch.nn as nn


class MyLayer(nn.module):
    def __init__(self, size_in):
        super.__init__()
        self.size = size_in
        weight = torch.Tensor(self.size)
        self.weight = nn.Parameter(weight)  # parameters
        torch.nn.init.normal_(self.weight, mean=0.0, std=1.0)  # initialize the params with a normal distribution

    def forward(self, x):
        return x * self.weight


"""
myNetwork = nn.Sequential(MyLayer(), nn.ReLU())
a = torch.ones([1, 10]) # 1- barch size 10- ne elem 
myNetwork(a)
"""