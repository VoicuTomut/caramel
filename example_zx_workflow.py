"""
PyZX and opt-einsum contraction workflow example:.
"""


from cotengra import ContractionTree
import numpy as np
import pyzx as zx
from carame_pyzx_enhancements.mcsim_tensor import mcsim_tensorfy

from caramel.interface_pyzx import Network
from caramel.path_optimizer.optimizer_mansikka import MansikkaOptimizer
from caramel.utils import contraction_moment, contraction_moment_to_zx_contraction

optimizer = MansikkaOptimizer()

# Opt-einsum with mansikka optimizer
# eq, shapes = opt_einsum.helpers.rand_equation(10, 3, seed=42)
# arrays = list(map(np.ones, shapes))
# path, path_info = opt_einsum.contract_path(eq, *arrays, optimize=optimizer)
# print("opt_einsum_path:", path)
# print("opt_einsum_path_info:", path_info.speedup)

circuit_path = "circuit_dataset/zx_circuits/000_test_circuit.qasm"
tensor_circuit = zx.Circuit.load(circuit_path)
zx_graph = tensor_circuit.to_graph()
zx.draw_matplotlib(zx_graph, labels=True, figsize=(8, 4), h_edge_draw='blue', show_scalar=False,
                   rows=None).savefig("figures/000_test_circuit.png")

quantum_net = Network(zx_graph)

print("\n ------ circuit network ------")
quantum_net.print_net()

contraction_path = optimizer(inputs=quantum_net.opt_einsum_input.copy(),
                             output=quantum_net.opt_einsum_output.copy(),
                             size_dict=quantum_net.size_dict.copy(),
                             memory_limit=500)

tree = ContractionTree.from_path(inputs=quantum_net.opt_einsum_input,
                                 output=quantum_net.opt_einsum_output,
                                 size_dict=quantum_net.size_dict,
                                 path=contraction_path)

print("\n ------ contraction cost summary ------")
print(
    "log10[FLOPs]: ",
    "%.3f" % np.log10(tree.total_flops()),  # sum of the flops cont by every nod ein the tree
    " log2[SIZE]: ",
    "%.0f" % tree.contraction_width(),  # log 2 of the size of the largest tensor
    " log2[WRITE]: ",
    "%.3f" % np.log2(tree.total_write()),  # total amount of created  memory
)
print("contraction_path:", contraction_path)
print("input nodes:", quantum_net.opt_einsum_input)
print("\n ------ contraction edge/step ------")
cm = contraction_moment(quantum_net.opt_einsum_input,
                        quantum_net.size_dict,
                        contraction_path)
print(cm)

print("\n ------ contraction in zx format ------")
contraction_list = contraction_moment_to_zx_contraction(cm, quantum_net)
print("contraction_list zx:", contraction_list)
print("nr edges to contract:", len(contraction_list))

custom_order_result = zx_graph.to_matrix(my_tensorfy=mcsim_tensorfy, contraction_order=contraction_list)
print("circuit matrix:\n", custom_order_result)

simple_zx_result = zx_graph.to_matrix()
print(" simple circuit matrix:\n", custom_order_result)
print("Equals:         ", np.allclose(custom_order_result, simple_zx_result))

s = 0
for i in range(len(custom_order_result)):
    for j in range(len(custom_order_result)):
        s = s + (abs(custom_order_result[i][j] - simple_zx_result[i][j])) ** 2
print("Difference between results :", s)
