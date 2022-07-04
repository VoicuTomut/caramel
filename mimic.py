"""

"""
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch_geometric
from caramel.models.dual_data_set_builder import CircuitDataset as DualCircuitDataset
from caramel.utils import node_colour_contraction

from caramel.models.m1_example import Dummy_Net

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\n-Device set!-\n\n")

# Get data set

tf = ['000_test_circuit.qasm', 'tof_10_after_heavy', 'tof_10_after_light', 'tof_10_before',
      'tof_10_pyzx.qc', 'tof_10_tpar.qc', 'tof_3_after_heavy', 'tof_3_after_light',
      'tof_3_before', 'tof_3_pyzx.qc', 'tof_3_tpar.qc', 'tof_4_after_heavy', 'tof_4_after_light',
      'tof_4_before', 'tof_4_pyzx.qc', 'tof_4_tpar.qc', 'tof_5_after_heavy', 'tof_5_after_light',
      'tof_5_before', 'tof_5_pyzx.qc', ]

dataset = DualCircuitDataset(root='C:/Users/tomut/Documents/GitHub/caramel/circuit_dataset/dual_experiment_dataset/',
                             target_files=tf)
print("dataset:", dataset)
print("\n-Data extracted!- \n\n")

data = dataset[0]
print("x_shape", dataset[0].x.shape[1])
print(data)
# color_map = node_colour_contraction(data, x_poz=2)
# g = torch_geometric.utils.to_networkx(data, to_undirected=True)
# nx.draw(g, with_labels=True, node_color=color_map)
# plt.savefig("figures/dual_input_graph.png")
# plt.close()

# color_map = node_colour_contraction(data, x_poz=None)
# nx.draw(g, with_labels=True, node_color=color_map)
# plt.savefig("figures/dual_target_graph.png")
# plt.close()


# Model
feature_size = dataset[0].x.shape[1]
model = Dummy_Net(feature_size=feature_size)
print("model:", model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

# Train
model = model.to(device)



data = dataset[0].to(device)
