"""
Architecture from https://github.com/raminass/protein-contrastive/blob/main/clip/architecture.py
"""

# Classes used for coembedding
import torch

import numpy as np


import torch.nn as nn


from torch.utils.data import Dataset


import torch.nn.functional as F


class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        return torch.from_numpy(sample).float()


class Protein_Dataset(Dataset):
    def __init__(self, data_train_x, data_train_y):
        self.data_train_x = data_train_x
        self.data_train_y = data_train_y

    def __len__(self):
        return len(self.data_train_x)

    def __getitem__(self, item):
        return self.data_train_x[item], self.data_train_y[item], item


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight.data)
        #   nn.init.kaiming_normal(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)


def init_weights_d(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight.data)


class ClipModel(nn.Module):
    def __init__(self, x_input_size, y_input_size, latent_dim, hidden_size, dropout):
        super().__init__()

        self.encoder_x = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(x_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Tanh(),
        )

        self.encoder_y = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(y_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Tanh(),
        )

        # tempature parameter for softmax to be learned
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # initialize weights
        self.encoder_x.apply(init_weights)
        self.encoder_y.apply(init_weights)

    def get_embedding(self, x, y):
        h_x = self.encoder_x(x)
        h_y = self.encoder_y(y)
        # normalize h_x and h_y
        e_x = F.normalize(h_x, p=2, dim=-1)
        e_y = F.normalize(h_y, p=2, dim=-1)

        return e_x, e_y

    def forward(self, x, y):
        e_x, e_y = self.get_embedding(x, y)

        return e_x, e_y, self.logit_scale.exp()
