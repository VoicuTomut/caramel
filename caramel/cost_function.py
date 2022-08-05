"""
This is an alteration of the contraction path that can be used as a cost function during learning
"""

import numpy as np
from cotengra import ContractionTree
from .utils import edge_path_to_opt_einsum_path


def cost_of_contraction(path, quantum_net, importance=[1, 1, 1]):
    """
    :path:
    :quantum_net:
    :importance: [ a, b, c] = [sum of the flops cont by every nod ein the tree,
                                log 2 of the size of the largest tensor,
                                total amount of created  memory]
    :return: total cost of contraction
    """
    path = edge_path_to_opt_einsum_path(path, quantum_net.opt_einsum_input.copy())
    tree = ContractionTree.from_path(inputs=quantum_net.opt_einsum_input,
                                     output=quantum_net.opt_einsum_output,
                                     size_dict=quantum_net.size_dict,
                                     path=path)

    cost = importance[0] * np.log10(tree.total_flops()) + importance[1] * tree.contraction_width() + importance[
        2] * np.log2(tree.total_write())

    return cost
