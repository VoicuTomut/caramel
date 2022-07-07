"""
MansikkaOptimizer  is an addaptation of the algorithm from the paper paper
"""

import copy
import opt_einsum as oem


class MansikkaOptimizer(oem.paths.PathOptimizer):

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        """

        :param inputs:
        :param output:
        :param size_dict:
        :param memory_limit:
        :return:
        """

        mansikka = MansikkaGraph(inputs, output, size_dict)
        contraction_order = get_contraction_order(mansikka, nr_tensors_to_rem=2,
                                                  nr_iter=10)  # [(0, 1)] * (len(inputs) - 1)
        contraction_order = contraction_order_to_opt_einsum(contraction_order, inputs, output)
        # print("contraction_order", contraction_order)

        return contraction_order


class MansikkaGraph:
    """
    the dual of the input graph
    """

    def __init__(self, inputs, output, size_dict):

        # print("sizedic :",size_dict)
        dual_vert = [k for k in size_dict.keys() if k not in output]

        dual_edges = set()
        for node in dual_vert:
            k = 0
            edge = [0, 0]
            for i, tensor in enumerate(inputs):
                if node in tensor:
                    edge[k] = i
                    k += 1
                    if k == 2:
                        dual_edges.add((min(edge), max(edge)))
                        break
        # print("len dual edges",len(dual_edges))
        # print("len dual ver:", len(dual_vert))
        self.vertices = dual_vert
        self.edges = dual_edges

    def neighbourhood(self, node):
        """
        Args:
            node:

        Returns: list of all the neighbors of the node node

        """
        nb = []

        for edge in self.edges:
            if node in edge:
                for v in edge:
                    if v != node and v not in nb:
                        nb.append(v)

        return nb

    def construct_dual(self):
        dual_vert = [v for v in self.edges]
        dual_edges = set()

        for d_node in dual_vert:
            n1 = d_node[0]
            n2 = d_node[1]
            for d_node2 in dual_vert:
                if d_node2 != d_node:
                    if n1 in d_node2 or n2 in d_node2:
                        if (d_node2, d_node) not in dual_edges:
                            dual_edges.add((d_node, d_node2))

        return MansikkaGraph(dual_vert, dual_edges)

    def find_treewidth_from_order(self, elimination_order):

        # Work on a copy
        working_graph = copy.deepcopy(self)
        # working_graph = Graph(self.vertices.copy(), self.edges.copy())
        treewidth = 1

        for u in elimination_order:

            nb = working_graph.neighbourhood(u)  # neighbourhood of vertices V

            for i in range(len(nb)):
                for j in range(i + 1, len(nb)):
                    if (nb[j], nb[i]) not in working_graph.edges:
                        working_graph.edges.add((nb[i], nb[j]))

            if len(working_graph.neighbourhood(u)) > treewidth:
                treewidth = len(working_graph.neighbourhood(u))

            working_graph.vertices.remove(u)
            edges = working_graph.edges.copy()

            for edge in edges:
                if u in edge:
                    working_graph.edges.remove(edge)

        return treewidth

    def greedy_treewidth_deletion(
            self, elimination_order, nr_tensors_to_rem, option=False, direct_minimization=False
    ):
        """
        Args:
            self:
            elimination_order: ??
            nr_tensors_to_rem: Parameter m from the paper
            option:
            direct_minimization:

        Returns:

        """
        removed_vertices = []
        new_graph = copy.deepcopy(self)
        # new_graph = Graph(self.vertices, self.edges)
        new_order = elimination_order.copy()

        for j in range(nr_tensors_to_rem):

            if direct_minimization:
                # if new order is given direct treewidth metric is used
                u = new_graph.removal_recommendation(new_order)
            else:
                u = new_graph.removal_recommendation()

            # update new graph
            new_graph.vertices.remove(u)

            edges = new_graph.edges.copy()
            for edge in edges:
                if u in edge:
                    new_graph.edges.remove(edge)

            # update removed_vertices
            if u not in removed_vertices:
                removed_vertices.append(u)

            # update new_order
            new_order.remove(u)
            if option is True:
                new_graph.tree_decomposition()

        tw = new_graph.find_treewidth_from_order(new_order)

        return new_graph, new_order, tw, removed_vertices

    def removal_recommendation(self, order=None):
        # this will be updated to betweenness centrality

        if order is not None:
            return self.direct_treewidth_minimization(order)

        nr_neighbors = 0
        recommendation = self.vertices[0]
        for v in self.vertices:
            nr_nb = len(self.neighbourhood(v))
            if nr_nb > nr_neighbors:
                recommendation = v
                nr_neighbors = nr_nb

        return recommendation

    def direct_treewidth_minimization(self, elimination_order):

        # copy
        working_graph = copy.deepcopy(self)
        # working_graph = Graph(tensor_graph.vertices.copy(), tensor_graph.edges.copy())

        tw = working_graph.find_treewidth_from_order(elimination_order.copy())
        delta = 0
        recommendation = self.vertices[0]

        for u in self.vertices:

            new_order = elimination_order.copy()
            new_order.remove(u)

            # TODO Alexandru: copy new_graph or working_graph?
            new_graph = copy.deepcopy(self)
            # new_graph = Graph(self.vertices.copy(), self.edges.copy())

            new_graph.vertices.remove(u)
            edges = new_graph.edges.copy()

            for edge in edges:
                if u in edge:
                    new_graph.edges.remove(edge)

            n_tw = new_graph.find_treewidth_from_order(new_order)

            new_delta = tw - n_tw

            if new_delta <= delta:
                delta = new_delta
                recommendation = u

        return recommendation

    def tree_decomposition(self):
        """
        Not implemented
        Args:
            self:

        Returns:

        """
        return None


def get_contraction_order(graph, nr_tensors_to_rem, nr_iter=10):
    working_graph = graph  # MansikkaGraph(graph.vertices.copy(), graph.edges.copy())
    initial_order = [k for k in graph.vertices]
    # return initial_order
    contraction_order = []

    initial_tw = working_graph.find_treewidth_from_order(initial_order)

    if initial_tw == 1:
        return initial_order

    while nr_iter > 0:
        nr_iter = nr_iter - 1
        reduced_g, reduced_order, tw, removing_order = \
            working_graph.greedy_treewidth_deletion(
                initial_order, nr_tensors_to_rem
            )
        for node in removing_order:
            contraction_order.append(node)

        initial_order = reduced_order.copy()
        working_graph = reduced_g
        if tw == 1:
            break

    for node in reduced_order:
        contraction_order.append(node)

    return contraction_order


def contraction_order_to_opt_einsum(contraction_order, inputs, outputs):
    # print("inputs", inputs)
    # print("outputs", outputs)
    # print("initial contraction order:", contraction_order)
    opt_einsum_contraction = []

    output_close_edges = []
    for edge in contraction_order:
        k = 0
        rm = [0, 0]
        output_node = None
        for i, node in enumerate(inputs):
            if edge in node:
                for out in outputs:
                    if out in node:
                        output_node = k,
                        out_saved = out
                rm[k] = i
                k += 1
                if k == 2:
                    if output_node:
                        output_close_edges.append(((rm[0], rm[1]), out_saved))
                    opt_einsum_contraction.append((rm[0], rm[1]))
                    break

    reorder_output_close_edges: list[tuple[int, int]] = []
    for edge in outputs:
        for pair in output_close_edges:
            if edge in inputs[pair[0][0]]:
                reorder_output_close_edges.append(pair)
            elif edge in inputs[pair[0][1]]:
                reorder_output_close_edges.append(pair)

    for i in range(len(reorder_output_close_edges)):
        edge = reorder_output_close_edges[i][0]
        # print(edge)
        # print("output:", reorder_output_close_edges[i][1])
        opt_einsum_contraction.append(edge)

    total = len(inputs)
    new_tensor = total
    name_list = {k: k for k in range(total)}
    valid_list = {k: k for k in range(total)}

    update_order = []
    sh = []
    for edge in opt_einsum_contraction:

        l0 = search_namelist(name_list, edge[0])
        l1 = search_namelist(name_list, edge[1])

        if valid_list[l0] != valid_list[l1]:

            update_order.append((valid_list[l0], valid_list[l1]))
            sh.append((edge[0], edge[1]))
            # Update validation
            mi = min(valid_list[l0], valid_list[l1])
            ma = max(valid_list[l0], valid_list[l1])
            del valid_list[l0]
            del valid_list[l1]
            valid_list[new_tensor] = total
            for k in valid_list.keys():
                if mi < valid_list[k] < ma:
                    valid_list[k] = valid_list[k] - 1
                elif valid_list[k] > ma:
                    valid_list[k] = valid_list[k] - 2

            name_list[l0] = new_tensor
            name_list[l1] = new_tensor
            name_list[new_tensor] = new_tensor
            new_tensor = new_tensor + 1
            total = total - 1

    return update_order


def search_namelist(name_list, name):
    # print("namelist:",name_list)
    start = name
    # print("0_start:",start)
    stop = name_list[start]
    # print("0_stop:",stop)
    while start != stop:
        start = stop
        # print("0_start:", start)
        stop = name_list[start]
        # print("1_stop:", stop)
    return start
