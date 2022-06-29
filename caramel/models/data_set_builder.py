import os

import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm

import pyzx as zx
from caramel.interface_pyzx import Network
from caramel.optimizer_mansikka import MansikkaOptimizer


class CircuitDataset(Dataset):
    def __init__(self, root, target_files=None, test=False, transform=None, pre_transform=None, pre_filter=None):
        """

        :param root: Where the data set should be stored. This folder is split in raw_dir and processed_dir
        :param transform:
        :param pre_transform:
        :param pre_filter:
        """
        self.test = test
        self.tf = target_files
        super(CircuitDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """
        If this files exist in the raw_dir download is not triggered.
        Download function is not implemented yet
        :return:
        """
        return self.tf

    def download(self):
        pass

    @property
    def processed_file_names(self):
        """
        If these files are already in processed_dir the processing is skipped.
        :return:
        """
        processed_file_names = []
        for file in self.tf:
            if file[-2] == ".":
                processed_file_names.append(file[::-2] + ".pc")
            else:
                processed_file_names.append(file[::-2] + ".pc")

        return processed_file_names

    def process(self):

        for raw_path in tqdm(self.raw_paths, total=len(self.raw_paths)):
            tensor_circuit = zx.Circuit.load(raw_path)
            zx_graph = tensor_circuit.to_graph()
            quantum_net = Network(zx_graph)

            nod_feats = self._get_node_feats(quantum_net)
            edge_feats = self._get_edge_feats(quantum_net)
            edge_index = self._get_connectivity(quantum_net)

            contraction_suggestion = self._get_additional_info(quantum_net)

            data = Data(x=nod_feats,
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=contraction_suggestion,
                        )

            file_name = raw_path.split("\\")[-1].split(".")[0]
            torch.save(data, os.path.join(self.processed_dir, f'{file_name}.pt'))

    def _get_node_feats(self, quantum_net):
        """
        Return a 2d tensor  of shape [number of nodes, node features size]
        :param quantum_net:
        :return: pytorch tensor
        """
        # degree -> number of edges
        # order  -> -1 for nodes that don't have a dangling edge
        #           or 'i' where 'i' is the position in the output_order

        all_node_feats = []
        for key in quantum_net.node_collection.keys():
            node = quantum_net.node_collection[key]
            node_feats = []
            degree = len(node["edges"])
            order = -1
            for edge in node["edges"]:
                for output_order in range(len(quantum_net.opt_einsum_output)):
                    ed = quantum_net.opt_einsum_output[output_order]
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
        :param quantum_net:
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

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.tf[idx]}.pt'))
        return data


"""
tf = ['tof_10_after_heavy', 'tof_10_after_light', 'tof_10_before',
      'tof_10_pyzx.qc', 'tof_10_tpar.qc', 'tof_3_after_heavy', 'tof_3_after_light',
      'tof_3_before', 'tof_3_pyzx.qc', 'tof_3_tpar.qc', 'tof_4_after_heavy', 'tof_4_after_light',
      'tof_4_before', 'tof_4_pyzx.qc', 'tof_4_tpar.qc', 'tof_5_after_heavy', 'tof_5_after_light',
      'tof_5_before', 'tof_5_pyzx.qc', ]

dataset = CircuitDataset(root='C:/Users/tomut/Documents/GitHub/caramel/circuit_dataset/experiment_dataset/',
                         target_files=tf)

idx = 6
print("edge_index:\n", dataset[6].edge_index)
print("node_atr:\n", dataset[6].x)
print("edge_atr:\n", dataset[6].edge_attr)
print("order\n", dataset[6].y)
"""