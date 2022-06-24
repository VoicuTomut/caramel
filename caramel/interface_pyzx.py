"""

"""


class Network:
    def __init__(self, zx_graph):
        self.node_collection = self.get_node_collection(zx_graph)
        self.opt_einsum_output = self.get_output(zx_graph)
        self.opt_einsum_input = self.get_input(zx_graph)
        self.size_dict = self.get_size_dic(zx_graph)

    def print_net(self):
        print("\n node_collection:", self.node_collection)
        print("\n input:", self.opt_einsum_input)
        print("\n output:", self.opt_einsum_output)
        print("\n size_dic:", self.size_dict)

    def get_node_collection(self, zx_graph):
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
            node_collection[node]["tensor"] = get_tensor()

        return node_collection

    def get_output(self, zx_graph):

        inputs = zx_graph.inputs()
        outputs = zx_graph.outputs()

        opt_einsum_output = []
        for i in range(len(inputs)):
            opt_einsum_output.append(*self.node_collection[inputs[i]]["edges"])
            opt_einsum_output.append(*self.node_collection[outputs[i]]["edges"])

        return opt_einsum_output

    def get_input(self, zx_graph):

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

        inputs = zx_graph.inputs()
        outputs = zx_graph.outputs()

        # inp = []
        # for circ_node in self.get_node_collection(zx_graph).keys():
        #     if circ_node not in inputs:
        #         if circ_node not in outputs:
        #             inp.append(self.node_collection[circ_node]["edges"])
        #
        # size_dic = {}
        # for i in range(len(inp)):
        #     size_dic[i] = len(inp[i])

        inp = []
        for circ_node in self.get_node_collection(zx_graph).keys():
            if circ_node not in inputs:
                if circ_node not in outputs:
                    inp.append(self.node_collection[circ_node]["edges"])

        size_dic = {}
        for node_edges in inp:
            for edge in node_edges:
                if edge in size_dic.keys():
                    size_dic[edge] = size_dic[edge]
                else:
                    size_dic[edge] = 1

        return size_dic


def get_tensor():
    return 0
