import numpy as np
from model import Net
from read_data import transform_data
from torch_geometric.loader import DataLoader
import torch
from utils import calc_loss, plot_loss, split_data, get_normalization
import os

# Constants
BATCH_SIZE = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Currently using ", device)

# Generate graph:
n_node = 5  # Number of nodes
in_node_channels = 4  # Node features: init_cars, upcoming cars, demands, bonuses
in_edge_channels = 2  # Edge features: distance, traffic flow
out_node_channels = n_node  # Represent the distribution propotion of idle drivers

# Define our dataset
path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/train_model/data_cat.npy"
data = np.load(path, allow_pickle=True)
model_save_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/train_model/"

# Normalization
normalization_config = get_normalization(data)
# data_list = transform_data(data, normalization_config, doNormalize=True)
data_list = transform_data(data, normalization_config, doNormalize=False)
# train_loader = DataLoader(datpaset=data_list, batch_size=BATCH_SIZE, shuffle=False)
train_data_list, validate_data_list = split_data(data_list, 10)
train_loader = DataLoader(dataset=train_data_list, batch_size=BATCH_SIZE, shuffle=True)
validate_loader = DataLoader(dataset=validate_data_list, batch_size=len(validate_data_list), shuffle=True)


# -----------------------------------------------------------------------------------------------------------
# Main loop for training
# -----------------------------------------------------------------------------------------------------------
from tqdm import tqdm

# Constants
MAX_EPOCH = 200
LR_RATE = 1e-4
EARLY_STOP_CNT = 10 # after 10 times little decrease on loss, stop training
# Saving model prefix
model_prefix = 'net_param_run_'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Currently using ", device)
config = {
    "in_channels": in_node_channels, 
    "edge_channels": in_edge_channels, 
    "out_channels": out_node_channels, 
    "dim_message": 8,
    "message_hidden_layer": [128, 64],
    "update_hidden_layer": [64,32],
    # "message_hidden_layer": 32,
    # "update_hidden_layer": 32,
}
model = Net(n_node, config, device, flow='source_to_target').to(device)

# If you need continue training, please uncomment following line and modify the file's name. 
# model_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/train_model/tran_model.pkl"
# model.load_state_dict(torch.load(model_path))

optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE, weight_decay=5e-4)

train_loss = []
validate_loss = []
early_stop_counter = 0
for epoch in range(MAX_EPOCH):
    loss = []
    model.train()
    for data in tqdm(train_loader):
    # for idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        result = model(data)
        lossc = calc_loss(result, data.y)
        loss.append(lossc.item())
        lossc.backward()

        optimizer.step()
    train_loss.append(np.mean(np.array(loss)))

    loss = []
    model.eval()
    for data in validate_loader:
        data = data.to(device)
        result = model(data)
        lossc = calc_loss(result, data.y)
        loss.append(lossc.item())
    validate_loss.append(np.mean(np.array(loss)))
    
    if (epoch+1) % 25 == 0:
        torch.save(model.state_dict(), model_save_path+model_prefix +str(epoch) +'.pkl')
    
    if (epoch+1) % 2==0:
        print('In epoch:', epoch, ' train loss:', train_loss[-1], " validate loss: ", validate_loss[-1])

    # If the performance increased, increment the early stop counter
    if epoch!=0 and np.floor(validate_loss[epoch])>=np.floor(validate_loss[epoch-1]):
        early_stop_counter+=1
    else:
        early_stop_counter = 0

    if early_stop_counter>=EARLY_STOP_CNT:
        print("Early stopped, since the performance did not improved in the last ", EARLY_STOP_CNT, " rounds. ")
        torch.save(model.state_dict(), model_save_path+"tran_model.pkl")
        break

# Plot the loss curve
plot_loss(train_loss, validate_loss)
print("In training, the final loss is: ", train_loss[-1])
print("In validation, the final loss is: ", validate_loss[-1])

print("Done")