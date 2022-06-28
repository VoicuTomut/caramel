import numpy as np
import pyzx as zx

from caramel.interface_pyzx import Network
from caramel.optimizer_mansikka import MansikkaOptimizer
from caramel.utils import *

circuit_path = "circuit_dataset/zx_circuits/000_test_circuit.qasm"
tensor_circuit = zx.Circuit.load(circuit_path)
zx_graph = tensor_circuit.to_graph()

quantum_net = Network(zx_graph)
quantum_net.print_net()
abj_mat = quantum_net.adjacent_matrix_enhanced()
print("\n Input:\n Circuit abj mat enhanced:\n", abj_mat)

optimizer = MansikkaOptimizer()
contraction_order = optimizer(quantum_net.opt_einsum_input,
                              quantum_net.opt_einsum_output,
                              quantum_net.size_dict)

print("\n Output:\n Contraction order:\n", contraction_order)

# Contraction order rules :
# size: len(quantum_net.opt_einsum_input)-1
# possible value: [ ...(<i, <=i),...]
#


