import os, sys
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm

sys.path.insert(1, sys.path[0] + '/..')
from src.misc.files import read_image


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

def pad_tensor(x_unpadded, pad_value = 0, pad_size = 64):
    x = x_unpadded
    return nn.functional.pad(x, (
            0, pad_size - x.shape[-1],
            0, pad_size - x.shape[-2], 
            0, pad_size - x.shape[-3]
        ), value = pad_value)

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
    epsilon = 0.00001
    return (volume - mean) / (std + epsilon)

def normalize_dir(image_dir, args):
    """
        Normalize the images in image_dir to be in the range [0, 1]

        Heavy load! Takes a long time to run!
    """ 
    means = []
    voxels = []

    for file in tqdm(os.listdir(image_dir)):
        img_id = file.split('-')[0]
        img_path = f"{args.data_dir}/raw/{args.split}/images/{img_id}-image.nii.gz"
        img_data, _ = read_image(img_path)
        means.append(np.mean(img_data, axis=0))
        voxels.append(np.size(np.flatten(img_data)))
    
    mean_data = sum(means * voxels)/sum(voxels), voxels

    variances = []
    for file in tqdm(os.listdir(image_dir)):
        img_id = file.split('-')[0]
        img_path = f"{args.data_dir}/raw/{args.split}/images/{img_id}-image.nii.gz"
        img_data, _ = read_image(img_path)
        variances.append(np.mean((img_data - mean_data)**2, axis=0))

    variances_weightedmean = sum(variances * voxels)/sum(voxels)

    std = np.sqrt(variances_weightedmean)
    epsilon = 0.0000001

    for file in tqdm(os.listdir(image_dir)):
        img_data = (img_data - mean_data) / (std + epsilon)
    
    #Now write it into a nice file that we can store.

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

def clipping_bones(volume):
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