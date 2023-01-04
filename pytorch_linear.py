# DEFINE your network, please search on website : "pytorch拟合sin函数"
# try other functions as well. Make sure you know how to:
# 1 Construct a network,
# 2 Train the network 

# TODO: An interesting fact is that when the range of data extends from [0, pi*2] to [0,100], the loss extensively increased. 
# Maybe due to the fact that np.sin function excess the fitting capability of current neural network. 

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class network(nn.Module):   # Define the neural network
    def __init__(self) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_layer = torch.nn.Sequential().to(self.device)

        self.hidden_layer.add_module("layer", nn.Linear(1, 128))
        self.hidden_layer.add_module("relu", nn.ReLU())
        self.hidden_layer.add_module("layer2", nn.Linear(128, 128))
        self.hidden_layer.add_module("relu2", nn.ReLU())
        self.hidden_layer.add_module("layer3", nn.Linear(128, 1))

        self.hidden_layer.to(self.device)
    
    def forward(self, x):
        x = self.hidden_layer(x)
        return x



device = 'cuda' if torch.cuda.is_available() else 'cpu'

x = np.linspace(0, 2*np.pi, 500).reshape([-1,1])
y = np.sin(x)
x = torch.FloatTensor(x).to(device)
y = torch.FloatTensor(y).to(device)

nn = network()
optimizer = optim.Adam(nn.parameters(), lr=1e-3)

for iter_cnt in range(1500):
    pred_y = nn(x)
    loss = F.mse_loss(pred_y, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Iter {}, the loss is {}".format(iter_cnt, loss))
    loss_numpy = loss.detach().cpu().numpy()
    if np.abs(loss_numpy)<=1e-3:
        break

data = np.linspace(0, 2*np.pi, 20).reshape([-1,1])
data_torch = torch.FloatTensor( data ).to(device)
data_pred = nn(data_torch)

plt.figure()
plt.plot(data, data_pred.detach().cpu().numpy(), label="prediciton")
plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), label="label")
plt.legend()
plt.show()

print("Done")