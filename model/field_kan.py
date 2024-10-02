import torch_geometric as pyg
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn.conv import GATv2Conv
from fastkan import FastKAN as KAN

class GraphKANField(torch.nn.Module):
    def __init__(self, num_features=3, hidden_size=500, target_size=100, num_edge_features=2, num_gnn_layers=15, kan_size=200, kan_layers=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.num_layers = num_gnn_layers
        self.kan_size = kan_size
        self.kan_layers = kan_layers

        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(self.num_features, self.hidden_size, edge_dim=num_edge_features))

        for _ in range(self.num_layers - 2):
            self.convs.append(GATv2Conv(self.hidden_size, self.hidden_size, edge_dim=num_edge_features))
        self.convs.append(GATv2Conv(self.hidden_size, self.hidden_size, edge_dim=num_edge_features))

        self.kan = KAN([self.kan_size]*self.kan_layers+ [self.target_size])

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.convs[0](x, edge_index, edge_attr=edge_attr)
        x = F.leaky_relu(x)
        # Apply GATConv layers with residual connections
        for i, conv in enumerate(self.convs[1:-1]):
            residual = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.leaky_relu(x)
            x = x + residual  # Skip connection
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        # Apply linear layers with residual connections
        kan_res = self.kan(x)
        return kan_res