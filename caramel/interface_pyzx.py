"""
Interface between pyzx and caramel module.
"""
import numpy as np

import sys

sys.path.append("..")
from carame_pyzx_enhancements.extract_tensor import get_tensor_from_g


class Network:
    def __init__(self, zx_graph):
        """

        :param zx_graph: zx graph object.
        """
        self.node_collection = self.get_node_collection(zx_graph)
        self.opt_einsum_output = self.get_output(zx_graph)
        self.opt_einsum_input = self.get_input(zx_graph)
        self.size_dict = self.get_size_dic(zx_graph)
        self.coo_mat = self.coo_matrix()

    def print_net(self):
        print("\n node_collection:", self.node_collection)
        print("\n input:", self.opt_einsum_input)
        print("\n output:", self.opt_einsum_output)
        print("\n size_dic:", self.size_dict)
        print("\n coo_mat:")
        for i in range(len(self.coo_mat[0])):
            print("         {}-{}".format(self.coo_mat[0][i], self.coo_mat[1][i]))

    def get_node_collection(self, zx_graph):
        """

        :param zx_graph: zx graph object.
        :return: { node: } a dictionary with the zx nodes.
        """
        nodes = zx_graph.vertices()

        edge_translation = {}
        edge_id = 0
        for edge in zx_graph.edge_set():
            edge_translation[edge] = edge_id
            edge_id = edge_id + 1

        node_collection = {}
        for node in nodes:
            node_collection[node] = {}
            node_collection[node]["id"] = node
            node_collection[node]["edges"] = {edge_translation[edge] for edge in zx_graph.incident_edges(node)}
            node_collection[node]["tensor"] = get_tensor(zx_graph, node)

        return node_collection

    def get_output(self, zx_graph):
        """
        Construct the opt_einsum output that satisfy the zx circuit
        :param zx_graph: zx graph object.
        :return: [ int , ]  opt_einsum output format  [edge_i, edge_j ..]
        """
        inputs = zx_graph.inputs()
        outputs = zx_graph.outputs()

        opt_einsum_output = []
        for i in range(len(inputs)):
            opt_einsum_output.append(*self.node_collection[inputs[i]]["edges"])
            opt_einsum_output.append(*self.node_collection[outputs[i]]["edges"])

        return opt_einsum_output

    def get_input(self, zx_graph):
        """

        Construct the opt_einsum input that satisfy the zx circuit
        :param zx_graph: zx graph object.
        :return: [ {int,int} , ]  opt_einsum input format  [{edge_i, edge_}, {edge_j ..]
        """

        inputs = zx_graph.inputs()
        outputs = zx_graph.outputs()

        inp = []

        for circ_node in self.get_node_collection(zx_graph).keys():
            input_output_flag = True
            if circ_node in inputs:
                input_output_flag = False
            if circ_node in outputs:
                input_output_flag = False
            if input_output_flag:
                inp.append(self.node_collection[circ_node]["edges"])

        return inp

    def get_size_dic(self, zx_graph):
        """

        :param zx_graph: zx graph object.
        :return: { edge:nr_connections ..}
        """
        inputs = zx_graph.inputs()
        outputs = zx_graph.outputs()

        inp = []
        for circ_node in self.get_node_collection(zx_graph).keys():
            if circ_node not in inputs:
                if circ_node not in outputs:
                    inp.append(self.node_collection[circ_node]["edges"])

        size_dic = {}
        for node_edges in inp:
            for edge in node_edges:
                if edge in size_dic.keys():
                    size_dic[edge] = size_dic[edge] + 1
                else:
                    size_dic[edge] = 1

        return size_dic

    def adjacent_matrix_opt(self):
        """

        :return: [nodes x  nodes ] adjacent_matrix_opt
        """
        abj_mat = np.zeros((len(self.opt_einsum_input), len(self.opt_einsum_input)))
        for i, node_i in enumerate(self.opt_einsum_input):
            for j, node_j in enumerate(self.opt_einsum_input):
                intersect = node_i.intersection(node_j)
                if len(intersect) != 0 and i != j:
                    abj_mat[i][j] = 1  # next(iter(intersect), None)
                    abj_mat[j][i] = abj_mat[i][j]
        return abj_mat

    def adjacent_matrix_enhanced_opt(self):
        """
        Adjacent matrix with extra info on main diagonal.
        :return: [nodes x  nodes ] adjacent_matrix_opt
        """
        abj_mat = self.adjacent_matrix_opt()

        for j, edge in enumerate(self.opt_einsum_output):
            for i, node_i in enumerate(self.opt_einsum_input):
                if edge in node_i:
                    abj_mat[i][i] = j
        return abj_mat

    def coo_matrix(self, direct=False):
        """

        :param direct:  Bool if True the reverse edge will be added. The doubles will be added at the end.
        :return:[[][]] connection matrix len(coo_mat[0])=len(coo_mat[1])=nr_edges.
        """

        coo_mat = [[0 for _ in range(len(self.size_dict))],
                   [0 for _ in range(len(self.size_dict))]]
        for edge in self.size_dict.keys():
            k = 0
            for i, key_i in enumerate(sorted(self.node_collection.keys())):
                node_i = self.node_collection[key_i]["edges"]
                if edge in node_i:
                    coo_mat[k][edge] = key_i

                    k = k + 1
            if k == 1:
                raise Exception("Some problem with coo mat  ")

        if direct:
            l = coo_mat[0]
            coo_mat[0] = coo_mat[0].append(coo_mat[1])
            coo_mat[1] = coo_mat[1].append(l)

        return coo_mat


def get_tensor(pyzx_graph, v):
    """
    Get tensor from the zx circuit.
    Not implemented yet because it wasn't required.
    :return: tensor.
    """

    return get_tensor_from_g(pyzx_graph, v)
