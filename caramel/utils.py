"""
Useful functions.
"""


def contraction_moment_to_zx_contraction(contraction_moments, circuit_net):
    """
    Convert a contraction moment list to zx contraction path
    :param contraction_moments: [ int, int,.. ] list of the moments on witch edges are contracted.
    :param circuit_net: a Network object
    :return:[zx_edge, zx_edge] py_zx contraction order. zx_edge=(node_a, node_b).
    """
    contraction = []
    coo_mat = circuit_net.coo_mat
    for moment in range(len(circuit_net.node_collection) - 1):
        for edge, edge_moment in enumerate(contraction_moments):
            if edge_moment == moment:
                contraction.append((coo_mat[0][edge], coo_mat[1][edge]))

    return contraction


def contraction_moment(opt_einsum_input, size_dic, contraction_path):
    """
    Convert opt_einsum contraction path to a moment list where moment[edge]= step on witch the edge is contracted.
    :param opt_einsum_input[ {edge_i, edge_j, edge_k},...]
    :param size_dic: { edge:nr_connections ...}
    :param contraction_path: [ (node_i,node_j), ..]
    :return: [ step_i, step_j, ...]
    """
    edge_moment = [len(opt_einsum_input) - 1 for _ in size_dic.keys()]

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
    """
    Return a coloring for the nodes from  data_graph
    :param data: PyTorch data graph
    :param x_poz: if true the coloring is based on node features else is base on the graph y.
    :return: [colour, colour,]  , colour in rgb format.
    """
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
    """
    Return a coloring for the edges from  data_graph
    :param data: PyTorch data graph
    :param edge_attr_number: if true the coloring is based on node features else is base on the graph y.
    :return: [colour, colour,]
    """
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
