"""
Useful functions.
"""


def edge_list_to_contraction_list(edge_list, opt_einsum_inputs):
    """
    convert a list of
    :param edge_list:
    :param opt_einsum_inputs:
    :return:
    """
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


def contraction_moment(opt_einsum_input, size_dic, contraction_path):
    """

    :param opt_einsum_input:
    :param size_dic:
    :param contraction_path:
    :return:
    """
    edge_moment = [len(opt_einsum_input)-1 for _ in size_dic.keys()]

    for moment, contraction in enumerate(contraction_path):
        contracted_edges = opt_einsum_input[contraction[0]].intersection(opt_einsum_input[contraction[1]])
        new_node = opt_einsum_input[contraction[0]].symmetric_difference(opt_einsum_input[contraction[1]])

        # update nodes
        if contraction[0] > contraction[1]:
            opt_einsum_input.pop(contraction[0])
            opt_einsum_input.pop(contraction[1])
        else:
            opt_einsum_input.pop(contraction[1])
            opt_einsum_input.pop(contraction[0])

        opt_einsum_input.append(new_node)

        for edge in contracted_edges:
            edge_moment[edge] = moment

    return edge_moment


def node_colour_contraction(data, x_poz=None):
    if x_poz is None:
        color_map = []
        for node in range(len(data.x)):
            x = (data.y[node]) / (max(data.y) + 0.0001)
            color_map.append((0.5, 0, x))

    else:
        y = []
        for i in range(len(data.x)):
            y.append(data.x[i][x_poz])
        color_map = []
        for node in range(len(data.x)):
            x = (y[node]) / (max(y) + 0.0001)
            color_map.append((0.5, 0, x))
    return color_map


def edge_colour_contraction(data, edge_attr_number=None):
    if edge_attr_number is None:
        color_map = []
        for node in range(len(data.edge_attr)):
            x = (data.y[node]) / (max(data.y) + 0.0001)
            color_map.append((0.5, 0, x))

    else:
        y = []
        for i in range(len(data.edge_attr)):
            y.append(data.edge_attr[i][edge_attr_number])
        color_map = []
        for node in range(len(data.edge_attr)):
            x = (y[node]) / (max(y) + 0.0001)
            color_map.append((0.5, 0, x))
    return color_map
