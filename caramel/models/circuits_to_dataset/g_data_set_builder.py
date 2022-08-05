"""
An example of how to create a pytorch data set from a set of quantum circuits.
In this case the circuit graph is not saved as it is. The data set contain the dual of the graph.
https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

dataset :
 x = dual_graph info
 y = MansikkaOptimizer contraction order
"""

import os

import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm

import pyzx as zx
from caramel.interface_pyzx import Network
from caramel.path_optimizer.optimizer_mansikka import MansikkaOptimizer
from caramel.utils import contraction_moment


class CircuitDataset(Dataset):
    def __init__(self, root, target_files=None, test=False, transform=None, pre_transform=None, pre_filter=None):
        """

        :param root: 'path/to/circuits' Where the data set should be stored.
                        This folder is split in raw_dir and processed_dir
        :param transform:  Optional  these are for the situation in witch we
                                     want to do additional changes to th edata set before processing
        :param pre_transform: Optional
        :param pre_filter: Optional
        """
        self.test = test
        self.tf = target_files
        super(CircuitDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """
        If this files exist in the raw_dir download is not triggered.
        Download function is not implemented yet
        :return: [ 'file_name'.. ]
        """
        return self.tf

    def download(self):
        pass

    @property
    def processed_file_names(self):
        """
        IIf these files are already in processed_dir the processing is skipped.
        :return: [ 'file_name'.. ]
        """
        processed_file_names = []
        for file in self.tf:
            if file[-2] == ".":
                processed_file_names.append(file[::-2] + ".pc")
            else:
                processed_file_names.append(file[::-2] + ".pc")

        return processed_file_names

    def process(self):
        """
        The core function for building the data set.
        """

        for raw_path in tqdm(self.raw_paths, total=len(self.raw_paths)):
            tensor_circuit = zx.Circuit.load(raw_path)
            zx_graph = tensor_circuit.to_graph()
            quantum_net = Network(zx_graph)

            nod_feats = self._get_node_feats(quantum_net)
            edge_index = self._get_connectivity(quantum_net)
            edge_feats = self._get_edge_feats(edge_index)

            contraction_evaluation_info = self._get_additional_info(quantum_net)

            data = Data(x=nod_feats,
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=contraction_evaluation_info,
                        )

            file_name = raw_path.split("\\")[-1].split(".")[0]
            torch.save(data, os.path.join(self.processed_dir, f'{file_name}.pt'))

    def _get_edge_feats(self, edg_index):
        """
        Return a 2d tensor  of shape [number of nodes, node features size]
        :param edg_index:  pytorch tensor. COO matrix
        :return: pytorch tensor
        """
        # For the dual representation edges represent only the fact that two
        # edges  from the initial circuit have a common node.

        all_node_feats = []
        for _ in range(len(edg_index[0])):
            all_node_feats.append([1])
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_node_feats(self, quantum_net):
        """
        Return a 2d tensor  of shape [number of edges, edge  features size]
        :param quantum_net:  a Network class object.
        :return: pytorch tensor
        """
        # degree -> number of nodes that connects in the circuit graph
        # order  -> -1 for edges that are not and edges
        #           or 'i' where 'i' is the position in the output_order
        # contraction_moment
        # contraction_moments for  k previous heuristics may be added in the future

        all_edge_feats = []
        for edge in quantum_net.size_dict:
            edge_feats = []

            degree = quantum_net.size_dict[edge]

            order = -1
            for output_order in range(len(quantum_net.opt_einsum_output)):
                ed = quantum_net.opt_einsum_output[output_order]
                if edge == ed:
                    order = output_order

            edge_feats.append(degree)
            edge_feats.append(order)
            edge_feats.append(len(quantum_net.opt_einsum_input) - 1)
            all_edge_feats.append(edge_feats)

        all_edge_feats = np.array(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_connectivity(self, quantum_net):
        """
         Return graph connectivity in COO format with shape [2, num_edges]
        :param quantum_net: a Network class object.
        :return:pytorch tensor
        """
        coo_mat = [[], []]
        # coo_mat = []
        nr_nodes = len(quantum_net.size_dict.keys())
        dual_abj = np.zeros((nr_nodes, nr_nodes))

        for k in quantum_net.size_dict.keys():
            for nodes in quantum_net.opt_einsum_input:
                if k in nodes:
                    for l in nodes:
                        dual_abj[k][l] = 1

        for j in range(0, len(dual_abj)):
            for k in range(j + 1, len(dual_abj)):
                if dual_abj[k][l] == 1:
                    # coo_mat.append([j, k])
                    # coo_mat.append([k, j])
                    coo_mat[0].append(j)
                    coo_mat[1].append(k)
                    coo_mat[0].append(k)
                    coo_mat[1].append(j)
        return torch.tensor(coo_mat)

    def _get_additional_info(self, quantum_net):
        """
        y -> the contraction order given by MansikkaOptimizer
        :param quantum_net: a Network class object
        :return: pytorch tensor
        """
        q_net = { 'opt_einsum_input': quantum_net.opt_einsum_input,
                  'opt_einsum_output': quantum_net.opt_einsum_output,
                  'size_dict': quantum_net.size_dict}


        return q_net

    def len(self):
        return len(self.processed_file_names)

    def get(self, id_x):
        if self.tf[id_x][-3] == ".":
            file = self.tf[id_x][:-3]
        elif self.tf[id_x][-5] == ".":
            file = self.tf[id_x][:-5]
        else:
            file = self.tf[id_x]

        data = torch.load(os.path.join(self.processed_dir, f'{file}.pt'))
        return data


"""
tf = ['000_test_circuit.qasm','tof_10_after_heavy', 'tof_10_after_light', 'tof_10_before',
      'tof_10_pyzx.qc', 'tof_10_tpar.qc', 'tof_3_after_heavy', 'tof_3_after_light',
      'tof_3_before', 'tof_3_pyzx.qc', 'tof_3_tpar.qc', 'tof_4_after_heavy', 'tof_4_after_light',
      'tof_4_before', 'tof_4_pyzx.qc', 'tof_4_tpar.qc', 'tof_5_after_heavy', 'tof_5_after_light',
      'tof_5_before', 'tof_5_pyzx.qc', ]

dataset = CircuitDataset(root='C:/Users/tomut/Documents/GitHub/caramel/circuit_dataset/dual_experiment_dataset/',
                         target_files=tf)

idx = -1
print("edge_index:\n", dataset[idx].edge_index)
print("node_atr:\n", dataset[idx].x)
print("edge_atr:\n", dataset[idx].edge_attr)
print("order\n", dataset[idx].y)
"""
