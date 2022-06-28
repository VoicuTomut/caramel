import os

import numpy as np
import torch
from torch_geometric.data import Dataset, Data

import pyzx as zx
from ..interface_pyzx import Network
from ..optimizer_mansikka import MansikkaOptimizer


class CircuitDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        """

        :param root: Where the data set should be stored. This folder is split in raw_dir and processed_dir
        :param transform:
        :param pre_transform:
        :param pre_filter:
        """
        super(CircuitDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """
        If this files exist in the raw_dir download is not triggered.
        Download function is not implemented yet
        :return:
        """
        return ['some_file_1', 'some_file_2', ...]

    def download(self):
        pass

    @property
    def process_file_name(self):
        """
        If these files are already in processed_dir the processing is skipped.
        :return:
        """
        return 'not_implemented.pt'

    def process(self):

        for raw_path in self.raw_paths:
            tensor_circuit = zx.Circuit.load(raw_path)
            zx_graph = tensor_circuit.to_graph()
            quantum_net = Network(zx_graph)

            nod_feats =  self._get_node_feats(quantum_net)
            edge_feats = self._get_edge_feats(quantum_net)
            edge_index = self._get_connectivity(quantum_net)

            contraction_suggestion = self._get_additional_info(quantum_net)

            data = Data(x=nod_feats,
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=contraction_suggestion,
                        )
            torch.save(data, os.path.join())

    def _get_node_feats(self, quantum_net):
        """
        Return a 2d tensor  of shape [number of nodes, node features size]
        :param path:
        :return: pytorch tensor
        """
        # degree -> number of edges
        # order  -> -1 for nodes that don't have a dangling edge
        #           or 'i' where 'i' is the position in the output_order

        all_node_feats = []
        for node in quantum_net.node_collection:
            node_feats = []
            degree = len(node["edges"])
            order = -1
            for edge in node["edges"]:
                for output_order in range(len(quantum_net.opt_einsum_output)):
                    ed = quantum_net.opt_einsum_output [output_order]
                    if edge == ed:
                        edge = output_order

            node_feats.append(degree)
            node_feats.append(order)
            all_node_feats.append(node_feats)

        all_node_feats = np.array(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_feats(self, quantum_net):
        """
        Return a 2d tensor  of shape [number of edges, edge  features size]
        :param path:
        :return: pytorch tensor
        """
        # degree -> number of nodes
        # order  -> -1 for edges that are not dangling edge
        #           or 'i' where 'i' is the position in the output_order
        # contraction_moment

        all_edge_feats = []
        for edge in quantum_net.size_dict:
            edge_feats = []

            degree = quantum_net.size_dict[edge]

            order = -1
            for output_order in range(len(quantum_net.opt_einsum_output)):
                ed = quantum_net.opt_einsum_output[output_order]
                if edge == ed:
                    order = output_order

            contraction_moment = 1

            edge_feats.append(degree)
            edge_feats.append(order)
            edge_feats.append(contraction_moment)
            all_edge_feats.append(edge_feats)

        all_edge_feats = np.array(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_connectivity(self, quantum_net):
        """
         Return graph connectivity in COO format with shape [2, num_edges]
        :param quantum_net:
        :return:
        """
        return quantum_net.coo_mat

    def _get_additional_info(self, quantum_net):

        optimizer = MansikkaOptimizer()
        contraction_order = optimizer(quantum_net.opt_einsum_input,
                                      quantum_net.opt_einsum_output,
                                      quantum_net.size_dict)
        return contraction_order