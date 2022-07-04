import opt_einsum
from opt_einsum import RandomGreedy

from cotengra import ContractionTree
import numpy as np
import pyzx as zx

from caramel.interface_pyzx import Network
from caramel.optimizer_mansikka import MansikkaOptimizer
from caramel.utils import contraction_moment

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
                             memory_limit = 500)

tree = ContractionTree.from_path(inputs=quantum_net.opt_einsum_input,
                                 output=quantum_net.opt_einsum_output,
                                 size_dict=quantum_net.size_dict,
                                 path=contraction_path)

print("\n ------ contraction cost summary ------")
print(
    "log10[FLOPs]: ",
    "%.3f" % np.log10(tree.total_flops()),
    " log2[SIZE]: ",
    "%.0f" % tree.contraction_width(),
    " log2[WRITE]: ",
    "%.3f" % np.log2(tree.total_write()),
)
print("contraction_path:",contraction_path)
print("input nodes:", quantum_net.opt_einsum_input)
print("\n ------ contraction edge/step ------")
cm = contraction_moment(quantum_net.opt_einsum_input,
                        quantum_net.size_dict,
                        contraction_path)
print(cm)



