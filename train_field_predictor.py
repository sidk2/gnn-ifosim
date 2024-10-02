import pickle
import time
import h5py

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import torch_geometric as pyg
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

from finesse.plotting.plot import get_2d_field
from finesse.utilities.homs import make_modes
from finesse.gaussian import BeamParam as BP

from model.field_kan import GraphKANField as FieldKAN

class IntensityDataset(Dataset):
    def __init__(self, data_files, max_size = None, transform=None, pre_transform=None):
        self.data_files = data_files
        self.data = []
        for j, file in enumerate(self.data_files):
            with h5py.File(file, 'r') as f:
                for i, key in enumerate(f.keys()):
                    if max_size and i > max_size[j]:
                        break
                    # Read and deserialize each graph
                    serialized_graph = f[key][()]
                    graph = pickle.loads(serialized_graph)
                    
                    data_tensor = pyg.utils.from_networkx(graph, group_node_attrs = ['Rc', 'R', 'alpha'], group_edge_attrs=['length', 'nr'])
                    data_tensor.x = torch.tensor(np.nan_to_num(data_tensor.x, posinf=0)).float()
                    data_tensor.y = torch.nan_to_num(self.get_intensity(data_tensor), posinf=0)
                    self.data.append(data_tensor)
        super(IntensityDataset, self).__init__(self.data, transform, pre_transform)

    def len(self):
        return len(self.data)
    
    def get_intensity(self, data):
        npts = 200
        modes = make_modes("even", maxtem=6)
        ints = torch.zeros((data.x.shape[0], int(npts/2)))
        for row in range(ints.shape[0]):
            _,_,amps = get_2d_field(modes, data.fd[row,:], BP(q=data.q[row].item()), samples=npts)
            intensity = (np.abs(amps)**2)[int(npts/2), int(npts/2):]
            ints[row, :] = torch.tensor(intensity).reshape(-1)/1000

        return ints
    
    def get(self, idx):
        # Load the data from the file
        data = self.data[idx]
        return data

def train(model, save_name, hyperparams):
    ''' 
    Train model with given hyperparams dict.

    Saves the following CSVs over the course of training:
    1. the loss trajectory: the val and train loss every save_loss_interval epochs at
       filename 'results/{name_prefix}_{learning_rate}_train.csv' e.g. 'results/baseline_0.05_train.csv'
    2. every save_model_interval save both the model at e.g. 'models/baseline_0.05_0_out_of_1000.pt`
       and the predicted values vs actual values in `results/baseline_0.05_0_out_of_1000_prediction.csv' on the test data.
    '''
    learning_rate = hyperparams['learning_rate']
    batch_size = hyperparams['batch_size']
    n_epochs = hyperparams['n_epochs']
    save_loss_interval = hyperparams['save_loss_interval']
    print_interval = hyperparams['print_interval']

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-2)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    losses = np.zeros((n_epochs, 3))
    
    min_val_loss = np.inf
    for epoch in range(n_epochs):
        epoch_loss = 0
        model.train()
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data).to(device)
            loss = nn.functional.l1_loss(out, data.y)
            epoch_loss += loss.item() 
            loss.backward()
            optimizer.step()
        if epoch % save_loss_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                tot_time = 0
                for dp in data_val:
                    dp = dp.to(device)
                    s = time.time()
                    out = model(dp).to(device)
                    e = time.time()
                    tot_time += e-s
                    val_loss += nn.functional.l1_loss(out.reshape(dp.y.shape).to(device), dp.y.to(device))
                val_loss /= len(data_val)
                train_loss = epoch_loss / len(data_train) * batch_size
            
                if val_loss < min_val_loss:
                    torch.save(model.state_dict(), f'{save_name}.pt')
                    min_val_loss = val_loss
                if epoch % print_interval == 0:
                    print("Epoch: {} Train loss: {:.2e} Validation loss: {:.2e}. Took {:.2e} seconds".format(epoch, train_loss, val_loss,e-s))
                    with open(f'{save_name}.txt', 'a+') as f:
                        f.write("Epoch: {} Train loss: {:.2e} Validation loss: {:.2e}, Mean Inference Time {:.4f}\n".format(epoch, train_loss, val_loss, tot_time / len(data_val)))
                losses[epoch, :] = (epoch, train_loss, val_loss.cpu())
        scheduler.step()
    return losses

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    dataset_path = 'gnn-ifosim/data/coupled_cavity_data.h5'

    hyperparams = {
        'batch_size' : 20, 
        'save_loss_interval' : 5, 
        'print_interval' : 1,
        'n_epochs' : 1000,
        'learning_rate' : 1e-4
    }
    
    dataset = IntensityDataset(data_files=[dataset_path])
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    test_size = dataset_size - train_size  # 20% for testing
    generator = torch.Generator().manual_seed(9301)
    
    # Split the dataset
    data_train, data_val = random_split(dataset, [train_size, test_size], generator=generator) 
    model = FieldKAN(target_size=100, num_features=3).to(device) # needs to be double precision
    model_loss_traj = train(model, "field_predictor", hyperparams)
