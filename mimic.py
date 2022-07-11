"""

"""
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from caramel.models.circuits_to_dataset.dual_data_set_builder import CircuitDataset as DualCircuitDataset

from caramel.models.dummy_model import DummyModel

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\n-Device set!-\n\n")

# Get data set

tf = ['000_test_circuit.qasm', 'tof_10_after_heavy', 'tof_10_after_light', 'tof_10_before',
      'tof_10_pyzx.qc', 'tof_10_tpar.qc', 'tof_3_after_heavy', 'tof_3_after_light',
      'tof_3_before', 'tof_3_pyzx.qc', 'tof_3_tpar.qc', 'tof_4_after_heavy', 'tof_4_after_light',
      'tof_4_before', 'tof_4_pyzx.qc', 'tof_4_tpar.qc', 'tof_5_after_heavy', 'tof_5_after_light',
      'tof_5_before', 'tof_5_pyzx.qc', ]

dataset = DualCircuitDataset(root='C:/Users/tomut/Documents/GitHub/caramel/circuit_dataset/dual_experiment_dataset/',
                             target_files=tf)
print("dataset:", dataset)
print("\n-Data extracted!- \n\n")

data = dataset[0]
print(data)
# color_map = node_colour_contraction(data, x_poz=2)
# g = torch_geometric.utils.to_networkx(data, to_undirected=True)
# nx.draw(g, with_labels=True, node_color=color_map)
# plt.savefig("figures/dual_input_graph.png")
# plt.close()

# color_map = node_colour_contraction(data, x_poz=None)
# nx.draw(g, with_labels=True, node_color=color_map)
# plt.savefig("figures/dual_target_graph.png")
# plt.close()


# Model
feature_size = dataset[0].x.shape[1]
model = DummyModel(feature_size=feature_size)
print("model:", model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

# untrained model
model = model.to(device)
data = dataset[0].to(device)

prediction = model(data.x, data.edge_index, data.edge_attr)
print("prediction:", prediction)


def mimic_loss(prediction, target):
    prediction = prediction.reshape(target.shape)
    # print(target)
    # print("max", max(target))
    max_t = max(target)
    # max_t = 1
    loss = torch.sum((prediction - target / max_t) ** 2)
    return loss


loss = mimic_loss(prediction, data.y)
print("loss:", loss)

# Training
nr_epochs = 500
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_hist = []
for epoch in range(nr_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for data in tqdm(dataset):
        # get the inputs; data is a list of [inputs, labels]
        sample = data.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        prediction = model(sample.x, sample.edge_index, sample.edge_attr)
        loss = mimic_loss(prediction, data.y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print(f'\n[{epoch + 1}] loss: {running_loss:.3f}')
    loss_hist.append(running_loss)

print('-Finished Training-\n ')

plt.plot(loss_hist)
plt.xlabel(' epochs')
plt.ylabel(' loss')
plt.title('loss history')
plt.savefig("figures/dummy_loss_history.png")
plt.show()
plt.close()

plt.plot(loss_hist[400:])
plt.xlabel(' epochs')
plt.ylabel(' loss')
plt.title('loss history cut ')
plt.savefig("figures/dummy_loss_history_cut.png")
plt.show()
plt.close()

# Model after training
data = dataset[0].to(device)
prediction = model(data.x, data.edge_index, data.edge_attr)
print("prediction:", prediction*max(data.y))
loss = mimic_loss(prediction, data.y)
print("loss:", loss)