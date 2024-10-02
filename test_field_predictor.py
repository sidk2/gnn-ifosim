import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import torch_geometric as pyg
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset

from train_field_predictor import IntensityDataset
from model.field_kan import GraphKANField as FieldKAN

if __name__ == '__main__':

    dataset = IntensityDataset(data_files=['gnn-ifosim/data/coupled_cavity_data.h5'], max_size=[100])
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    test_size = dataset_size - train_size  # 20% for testing

    # Split the dataset
    generator = torch.Generator().manual_seed(9301)
    data_train, data_val = random_split(dataset, [train_size, test_size], generator=generator) 
    model = FieldKAN(target_size=100, num_features=3) # needs to be double precision
    
    model.load_state_dict(torch.load('field_train_results/radial_mlp/field_fp.pt', weights_only=True, map_location=torch.device('cpu')))
    num_imgs = 10
    npts=200
    min_sum = np.inf
    min_ind = 0
    for i, data in enumerate(data_val):
        dp = model(data_val[i]).detach()
        dt = data_val[i].y.detach()
        print(torch.mean(data_val[i].y))
        if not i % 100:
            print(i)
        tot = torch.sum(dp - dt)
        if tot < min_sum:
            min_ind = i
    print(min_ind)
            
        
        
    
    loss = 0
    min_loss = np.inf
    
    ind = min_ind
    dp = model(data_val[ind])
    
    fig, axes = plt.subplots(num_imgs, 3, figsize=(14, 40))
    # print(data_val[ind].fd)
    for node_ind in range(num_imgs):
        r_values = np.linspace(0, 5, 100)
        f_r = model(data_val[ind])[node_ind].detach().numpy().reshape(-1)
        
        # Create a 2D grid of (x, y) values
        grid_size = 200  # size of the grid (200x200)
        x = np.linspace(-5, 5, grid_size)
        y = np.linspace(-5, 5, grid_size)
        X, Y = np.meshgrid(x, y)

        # Calculate radial distance r from the origin for each (x, y) point
        R = np.sqrt(X**2 + Y**2)

        # Interpolate the 1D f(r) values onto the 2D grid based on the radial distance R
        dp = np.interp(R, r_values, f_r)
        
        l0_em = dp.reshape((npts,npts))
        
        r_values = np.linspace(0, 5, 100)
        f_r = data_val[ind].y[node_ind].detach().numpy().reshape(-1)
        
        # Create a 2D grid of (x, y) values
        grid_size = 200  # size of the grid (200x200)
        x = np.linspace(-5, 5, grid_size)
        y = np.linspace(-5, 5, grid_size)
        X, Y = np.meshgrid(x, y)

        # Calculate radial distance r from the origin for each (x, y) point
        R = np.sqrt(X**2 + Y**2)

        # Interpolate the 1D f(r) values onto the 2D grid based on the radial distance R
        dt = np.interp(R, r_values, f_r)
        diff = dt-l0_em

        vmin=min([np.min(dt), np.min(l0_em), np.min(0)])
        vmax=max([np.max(dt), np.max(l0_em), np.max(0)])
        # Plot the first matrix
        im1 = axes[node_ind, 0].imshow(l0_em, cmap='viridis', vmin=np.min(l0_em), vmax=np.max(l0_em),interpolation='none')
        axes[node_ind, 0].set_title('Model')

        # Plot the second matrix
        im2 = axes[node_ind, 1].imshow(dt, vmin=np.min(dt), vmax=np.max(dt),cmap='viridis', interpolation='none')
        axes[node_ind, 1].set_title('Ground Truth')

        im3 = axes[node_ind, 2].imshow(diff, vmin=np.min(diff), vmax=np.max(diff),cmap='viridis', interpolation='none')
        axes[node_ind, 2].set_title('Difference')

        # Ensure both plots share the same colorbar
        divider = make_axes_locatable(axes[node_ind, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax, label='Intensity (W/m^2)')

        divider = make_axes_locatable(axes[node_ind, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax, label='Intensity (W/m^2)')

        divider = make_axes_locatable(axes[node_ind, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im3, cax=cax, label='Intensity (W/m^2)')

    plt.savefig("Image.png")
    plt.tight_layout()
    
r_values = np.linspace(0, 5, 100)
f_r = model(data_val[ind])[0].detach().numpy().reshape(-1)

plt.figure()
plt.plot(f_r)
plt.savefig('radial.png')