import pickle
import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import torch_geometric as pyg

from model.power_predictor import PowerMLP

class LazyDataset(Dataset):
    def __init__(self, data_files, max_size = None):
        self.data_files = data_files
        self.data = []
        for fn, file in enumerate(self.data_files):
            with h5py.File(file, 'r') as f:
                for i, key in enumerate(f.keys()):
                    if max_size and i > max_size[fn]:
                        break

                    # Read and deserialize each graph
                    serialized_graph = f[key][()]
                    graph = pickle.loads(serialized_graph)
                    self.data.append(graph)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the data from the file
        data = self.data[idx]
        data = pyg.utils.from_networkx(data, group_node_attrs = ['Rc', 'R', 'alpha'], group_edge_attrs=['length', 'nr'])
        data.x = torch.nan_to_num(data.x, posinf=0).float()
        data.y = torch.nan_to_num(torch.clamp(torch.log(data.pd), min=-10),posinf=0).float()
        
        return data.x.reshape(-1), data.y.reshape(-1)

def evaluate_model(model, data_iter):
    '''
    Accumulate MAE over a series of input graphs.
    '''
    return sum([nn.functional.l1_loss(model(data).reshape(data.y.shape), data.y).item() for data in data_iter])

def crit(mod, gt, lam, adj_matr):
    
    triu = torch.triu(adj_matr)
    tril = torch.tril(adj_matr)
    return nn.functional.l1_loss(gt, mod) + lam*(torch.log(torch.sum(torch.abs(triu@torch.exp(mod) - tril@torch.exp(mod)))))
def train(model, name_prefix, hyperparams, save_name):
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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)

    loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    min_val_loss = np.inf
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        model.train()
        s=time.time()
        for data, label in loader:
            data = data.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            out = model(data).to(device)
            loss = F.l1_loss(out.reshape(label.shape), label)
            epoch_loss += loss.item() 
            loss.backward()
            optimizer.step()
            
        e = time.time()
        if epoch % save_loss_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for dp, label in data_val:
                    dp = dp.to(device)
                    label = label.to(device)
                    out = model(dp).to(device)                
                    val_loss += F.l1_loss(out.reshape(label.shape), label)
                val_loss /= len(data_val)
                train_loss = epoch_loss / len(data_train) * batch_size
            
                if val_loss < min_val_loss:
                    torch.save(model.state_dict(), f'gnn-ifosim/results/power_train_results/{save_name}.pt')
                    min_val_loss = val_loss
                if epoch % print_interval == 0:
                    print("Epoch: {} Train loss: {:.2e} Validation loss: {:.2e}. Took {} seconds".format(epoch, train_loss, val_loss, e-s))
                    with open(f'gnn-ifosim/results/power_train_results/{save_name}.txt', 'a+') as f:
                        f.write("Epoch: {} Train loss: {:.2e} Validation loss: {:.2e}\n".format(epoch, train_loss, val_loss))
    return None

if __name__ == '__main__':
    hyperparams = {
        'batch_size' : 100, 
        'save_loss_interval' : 5, 
        'print_interval' : 1,
        'n_epochs' : 500,
        'learning_rate' : 1e-5
    }
    device = torch.device('cpu')
    
    generator = torch.Generator().manual_seed(9302)
    # Train a model with only FP data
    dataset = LazyDataset(data_files=['gnn-ifosim/data/fabry_perot_data.h5'], max_size=[100])
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    test_size = dataset_size - train_size  # 20% for testing

    # Split the dataset
    # print("Loaded dataset. Beginning training.")
    data_train, data_val = random_split(dataset, [train_size, test_size], generator=generator) 
    dp, lab = data_train[0]
    model = (PowerMLP( input_size = dp.shape[0], hidden_size=1000, num_layers=25, target_size = lab.shape[0])).to(device)
    model_loss_traj = train(model, "model", hyperparams, save_name='power_fp_mlp')
    
    generator = torch.Generator().manual_seed(9302)
    # Train a model with only half_aligo data
    dataset = LazyDataset(data_files=['gnn-ifosim/data/half_aligo_many_tops.h5'])

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    test_size = dataset_size - train_size  # 20% for testing

    # Split the dataset
    data_train, data_val = random_split(dataset, [train_size, test_size], generator=generator) 
    print("Loaded dataset. Beginning training.")
    dp, lab = data_train[0]

    model = (PowerMLP( input_size = dp.shape[0], hidden_size=1000, num_layers=25, target_size = lab.shape[0])).to(device) # needs to be double precision
    model_loss_traj = train(model, "model", hyperparams, save_name='power_aligo_mlp')

    
    
