

import cotengra as ctg
from cotengra import ContractionTree
import numpy as np
import pyzx as zx
from  opt_einsum import RandomGreedy

from carame_pyzx_enhancements.mcsim_tensor import mcsim_tensorfy
from caramel.interface_pyzx import Network
from caramel.path_optimizer.optimizer_mansikka import MansikkaOptimizer
from caramel.utils import contraction_moment, contraction_moment_to_zx_contraction

# optimizer = RandomGreedy()
# optimizer = MansikkaOptimizer()


tf = ['000_test_circuit.qasm', 'tof_10_after_heavy', 'tof_10_after_light', 'tof_10_before',
      'tof_10_pyzx.qc', 'tof_10_tpar.qc', 'tof_3_after_heavy', 'tof_3_after_light',
      'tof_3_before', 'tof_3_pyzx.qc', 'tof_3_tpar.qc', 'tof_4_after_heavy', 'tof_4_after_light',
      'tof_4_before', 'tof_4_pyzx.qc', 'tof_4_tpar.qc', 'tof_5_after_heavy', 'tof_5_after_light',
      'tof_5_before', 'tof_5_pyzx.qc', ]

folder_path = "circuit_dataset/zx_circuits/"
importance = [0, 0.1, 1]

loss = 0
for circuit in tf:
    print("circuit:", circuit)
    optimizer = RandomGreedy()
    circuit_path = folder_path+circuit
    tensor_circuit = zx.Circuit.load(circuit_path)
    zx_graph = tensor_circuit.to_graph()
    zx.draw_matplotlib(zx_graph, labels=True, figsize=(8, 4), h_edge_draw='blue', show_scalar=False,
                       rows=None).savefig("figures/000_test_circuit.png")

    quantum_net = Network(zx_graph)

    contraction_path = optimizer(inputs=quantum_net.opt_einsum_input.copy(),
                                 output=quantum_net.opt_einsum_output.copy(),
                                 size_dict=quantum_net.size_dict.copy(),
                                 memory_limit=500)
    # print(contraction_path)

    quantum_net = Network(zx_graph)
    tree = ContractionTree.from_path(inputs=quantum_net.opt_einsum_input,
                                     output=quantum_net.opt_einsum_output,
                                     size_dict=quantum_net.size_dict,
                                     path=contraction_path)

    cost = importance[0] * tree.total_flops() + importance[1] * np.log2(tree.contraction_width() + 1.0) + importance[
        2] * np.log2(float(tree.total_write() + 0.1))
    loss = loss+cost


    print(
        "log10[FLOPs]: ",
        "%.3f" % np.log10(tree.total_flops()),  # sum of the flops cont by every nod ein the tree
        " log2[SIZE]: ",
        "%.0f" % tree.contraction_width(),  # log 2 of the size of the largest tensor
        " log2[WRITE]: ",
        "%.3f" % np.log2(tree.total_write()),  # total amount of created  memory
    )

print("Total loss:", loss )