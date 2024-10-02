import numpy as np

import torch
from torch.utils.data import random_split

from train_power_predictor import PowerDataset
from model.power_predictor import LinGNN as PowerGNN

if __name__ == "__main__":
    dataset_path = 'gnn-ifosim/data/coupled_cavity_data.h5'
    model_path = 'power_train_results/coupled_cavity_trained/power_mixed_lin.pt'
    
    dataset = PowerDataset(data_files=[dataset_path])
    generator = torch.Generator().manual_seed(9302)
    
    dataset_size = len(dataset)
    train_size = int(0.8*dataset_size)  # 80% for training
    test_size = dataset_size - train_size  # 20% for testing

    # Split the dataset
    data_train, data_val = random_split(dataset, [train_size, test_size], generator=generator)
    
    model = PowerGNN()
    model.load_state_dict(torch.load(model_path))

    res = np.zeros((len(data_val), data_val[0].y.shape[0]))
    gt = np.zeros(res.shape)

    model.eval()
    
    with torch.no_grad():
        for ind, data in enumerate(data_val):
            gt[ind, :] = torch.exp(data.y)
            res[ind, :] = torch.exp(model(data).reshape(-1))
            
    print(np.mean(np.abs(gt-res)))