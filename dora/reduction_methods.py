import torch


def get_mean_along_last_2_dims(activations):
    return torch.mean(activations, dim=(2, 3))
