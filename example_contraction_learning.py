"""
Learning a contraction heuristic workflow example :
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
from caramel.models.circuits_to_dataset.g_data_set_builder import CircuitDataset as DualCircuitDataset
from caramel.cost_function import cost_of_contraction
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
print("prediction shape:", prediction.shape)




def contraction_loss(predictions, graphs_info):
    # path = predictions.reshape(predictions.shape[0])
    # print("predictions", predictions)
    # print("predictions shape", predictions.shape)
    predictions = predictions.reshape((1, predictions.shape[0]))
    contraction_cost_list = torch.tensor([[0.0]], dtype=torch.float32,requires_grad = True)
    for predicted_path in predictions:
        path = predicted_path
        contraction_cost = cost_of_contraction(path, graphs_info, importance=[0, 0.1, 1])
        contraction_cost_list = torch.cat([contraction_cost_list, torch.tensor([[contraction_cost]])], dim=1)
    # print("contraction cost list", contraction_cost_list)
    loss = torch.sum(contraction_cost_list)

    return loss


print("target:", data.y)
loss = contraction_loss(prediction, data.y)
print("loss:", loss)


print("###########Training##############")
# Training
nr_epochs = 50
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
        loss = contraction_loss(prediction, data.y)
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
plt.savefig("figures/loss_history.png")
plt.show()
plt.close()

plt.plot(loss_hist[0:])
plt.xlabel(' epochs')
plt.ylabel(' loss')
plt.title('loss history cut ')
plt.savefig("figures/loss_history_cut.png")
plt.show()
plt.close()

# Model after training
data = dataset[0].to(device)
prediction = model(data.x, data.edge_index, data.edge_attr)
print("prediction:", prediction * max(data.y))
loss = contraction_loss(prediction, data.y)
print("loss:", loss)
