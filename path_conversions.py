"""
PyZX and opt-einsum contraction workflow example:
"""


from cotengra import ContractionTree
import numpy as np
import pyzx as zx


from caramel.interface_pyzx import Network
from caramel.path_optimizer.optimizer_mansikka import MansikkaOptimizer
from caramel.utils import contraction_moment, contraction_moment_to_zx_contraction, edge_path_to_opt_einsum_path

optimizer = MansikkaOptimizer()


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

# print("\n ------ contraction cost summary ------")
# print(
#     "log10[FLOPs]: ",
#     "%.3f" % np.log10(tree.total_flops()),  # sum of the flops cont by every nod ein the tree
#     " log2[SIZE]: ",
#     "%.0f" % tree.contraction_width(),  # log 2 of the size of the largest tensor
#     " log2[WRITE]: ",
#     "%.3f" % np.log2(tree.total_write()),  # total amount of created  memory
# )

print("\n ------ contraction in opt-einsum format ------")
print("contraction_path:", contraction_path)
print("input nodes:", quantum_net.opt_einsum_input)
clone = quantum_net.opt_einsum_input.copy()

print("\n ------ contraction edge/step ------")
cm = contraction_moment(quantum_net.opt_einsum_input.copy(),
                        quantum_net.size_dict.copy(),
                        contraction_path)
print("contraction_moment:",cm)

print("\n ------ contraction in zx format ------")
contraction_list = contraction_moment_to_zx_contraction(cm, quantum_net)
print("contraction_list zx:", contraction_list)

print("\n ------ contraction_moment back to opt-einsum format ------")
b_path = edge_path_to_opt_einsum_path(cm, clone)
print("back opt-einsum:", b_path)

tree = ContractionTree.from_path(inputs=quantum_net.opt_einsum_input,
                                 output=quantum_net.opt_einsum_output,
                                 size_dict=quantum_net.size_dict,
                                 path=b_path)

print("\n ------ contraction cost summary ------")
print(
    "log10[FLOPs]: ",
    "%.3f" % np.log10(tree.total_flops()),  # sum of the flops cont by every nod ein the tree
    " log2[SIZE]: ",
    "%.0f" % tree.contraction_width(),  # log 2 of the size of the largest tensor
    " log2[WRITE]: ",
    "%.3f" % np.log2(tree.total_write()),  # total amount of created  memory
)
# print("nr edges to contract:", len(contraction_list))