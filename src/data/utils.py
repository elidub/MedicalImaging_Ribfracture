import torch
import numpy as np
from torch import nn

def crop(x, l = -1):
    x = x.unsqueeze(1) # Unsqueeze the channel dimension
    assert len(x.shape) == 5, "Input tensor must have 5 dimensions"
    if l != -1: x = x[:, :, :l, :l, :l] # Crop the input tensor if l is specified
    return x

def pad_list(x_unpadded, padding_value = 0, size = 64):
    return torch.stack([nn.functional.pad(x, (
            0, size - x.shape[-1],
            0, size - x.shape[-2], 
            0, size - x.shape[-3]
        ), value = padding_value) for x in x_unpadded])

def pad_tensor(x_unpadded, padding_value = 0, size = 64):
    x = x_unpadded
    return nn.functional.pad(x, (
            0, size - x.shape[-1],
            0, size - x.shape[-2], 
            0, size - x.shape[-3]
        ), value = padding_value)

def label_processor(self, y):
    # This is a placeholder function, such that it works with the dummy network!!!
    y = y.float()
    y = y[:, :, :, 0]  # Only select the first slice, similar to the dummy network
    y_downsampled = torch.nn.functional.avg_pool2d(y, kernel_size = 2*7, stride = 2**7).view(-1, 16)
    return y_downsampled

def normalize_standard(volume):
    """
        Normalize the volume to be in the range [0, 1]

        volume: np.array
    """
    mean = volume.mean()
    std = volume.std()
    return (volume - mean) / std

def normalize_minmax(volume):
    """
        Normalize the volume to be in the range [0, 1]

        volume: np.array
    """
    min_value = volume.min()
    max_value = volume.max()
    return (volume - min_value) / (max_value - min_value)

def simplify_labels(labels):
    """
        Project all label annotations which are not 0 to 1.

        labels: np.array
    """
    labels[labels != 0] = 1
    return labels

def extrapolate_bones(volume):
    """
        Remove low intensity voxels below intensity 200

        volume: np.array
    """
    return np.where(np.asarray(volume) > 200, volume, 0)

def clip_values(volume, min_value, max_value):
    """
        Clip the values of the volume to be in the range [min_value, max_value]

        volume: np.array
        min_value: int
        max_value: int
    """
    return np.clip(volume, min_value, max_value)