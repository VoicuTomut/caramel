"""
Useful functions.
"""


########################################################################################
# Some path conversions

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


def edge_path_to_opt_einsum_path(path, opt_einsum_input):
    # give to each edge a moment(int) at witch it will be contracted.
    edge_contraction = edge_contraction_path_to_dic(path)
    path = sorted(path)
    for moment, value in enumerate(path):
        for key in edge_contraction.keys():
            if edge_contraction[key] == value:
                edge_contraction[key] == moment

    edge_contraction = {k: v for k, v in sorted(edge_contraction.items(), key=lambda item: item[1])}

    opt_einsum_contraction = []  # [(node1,node5), (node2,node3)...]
    for edge in edge_contraction:

        contracted_nodes = []
        k = 0
        # print("opt-einsum_inp:",opt_einsum_input)
        for node_index, node in enumerate(opt_einsum_input):
            # print("edge {}-{}".format(edge, node))
            if edge in node:
                contracted_nodes.append(node_index)
                k = k + 1
            if k == 2:
                break

        if k == 2:
            new_node = opt_einsum_input[contracted_nodes[0]].symmetric_difference(opt_einsum_input[contracted_nodes[1]])
            opt_einsum_contraction.append((contracted_nodes[0], contracted_nodes[1]))
            opt_einsum_input.pop(contracted_nodes[0])
            if contracted_nodes[0]<contracted_nodes[1]:
                contracted_nodes[1] =contracted_nodes[1] -1
            opt_einsum_input.pop(contracted_nodes[1])
            opt_einsum_input.append(new_node)

    return opt_einsum_contraction


def edge_contraction_path_to_dic(path):
    """
    It creates a dictionary edge:{contraction moment number}
    """
    edge_contraction = {}
    for edge, moment in enumerate(path):
        edge_contraction[edge] = moment
    return edge_contraction


############################################################################
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
