import torch
import numpy as np


def rbf_torch(x1, x2, lengthscale):
    return torch.exp(-torch.sum((x1[:, None, :] - x2[None, :, :]) ** 2, dim=-1) / lengthscale ** 2)


def rbf_numpy(x1, x2, lengthscale):
    return np.exp(-np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1) / lengthscale ** 2)
