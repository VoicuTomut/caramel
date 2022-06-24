

import opt_einsum
from cotengra import ContractionTree
import numpy as np
import pyzx as zx

from caramel.interface_pyzx import Network
from caramel.optimizer_mansikka import MansikkaOptimizer


optimizer = MansikkaOptimizer()

# Opt-einsum with mansikka optimizer
eq, shapes = opt_einsum.helpers.rand_equation(10, 3, seed=42)
arrays = list(map(np.ones, shapes))
path, path_info = opt_einsum.contract_path(eq, *arrays, optimize=optimizer)
print("opt_einsum_path:",path)
print("opt_einsum_path_info:",path_info.speedup)


circuit_path = "circuit_dataset/zx_circuits/000_test_circuit.qasm"
tensor_circuit = zx.Circuit.load(circuit_path)
zx_graph=tensor_circuit.to_graph()


quantum_net = Network(zx_graph)
contraction_path = optimizer(inputs = quantum_net.opt_einsum_input,
                             output= quantum_net.opt_einsum_output,
                             size_dict= quantum_net.size_dict )
tree = ContractionTree.from_path(quantum_net.opt_einsum_input,
                                 quantum_net.opt_einsum_output,
                                 quantum_net.size_dict,
                                 path=contraction_path)

print("------ contraction cost summary ------")
print(
    "log10[FLOPs]: ",
    "%.3f" % np.log10(tree.total_flops()),
    " log2[SIZE]: ",
    "%.0f" % tree.contraction_width(),
    " log2[WRITE]: ",
    "%.3f" % np.log2(tree.total_write()),
)

