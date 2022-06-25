"""

"""


def edge_list_to_contraction_list(edge_list, opt_einsum_inputs):
    order = []
    nr_nodes = len(opt_einsum_inputs)
    for edge in edge_list:
        n = [0, 0]
        j = 0
        for i in range(nr_nodes):
            node = opt_einsum_inputs[i]
            if edge in node:
                n[j] = i
                j = j + 1
                if j == 2:
                    order.append(n)
                    opt_einsum_inputs.append(opt_einsum_inputs[n[0]], opt_einsum_inputs[n[1]])
                    del opt_einsum_inputs[i]
                    break
    return order
