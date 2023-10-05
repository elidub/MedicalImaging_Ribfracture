import torch
import pandas as pd
import sys, os
import nibabel as nib
import gzip
from PIL import Image
import numpy as np

sys.path.insert(1, sys.path[0] + '/..')
from src.data.utils import pad_tensor


def read_image(file_path):
    nii_img = nib.load(file_path)
    image_data = nii_img.get_fdata()
    header = nii_img.header
    return image_data, header

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, split='val', dir = '../data_dev', transform=None, target_transform=None):
        self.split = split
        self.split_dir = os.path.join(dir, split)

        self.transform = transform
        self.target_transform = target_transform

        self.IDs = []
        for filename in os.listdir(os.path.join(self.split_dir, 'images')):
            if filename.endswith(".nii.gz"):
                self.IDs.append(filename.split('-')[0])

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        ID = self.IDs[idx]

        img_path = os.path.join(self.split_dir, 'images', f"{ID}-image.nii.gz")
        image, _ = read_image(img_path)
        if self.transform: image = self.transform(image)
        
        if self.split == 'test':
            return image, -1 # return -1 as dummy label for test set
    
        labels_path = os.path.join(self.split_dir, 'labels', f"{ID}-label.nii.gz")
        label, _ = read_image(labels_path)
        if self.target_transform: label = self.target_transform(label)

        return image, label
    
class BoxesDataset(torch.utils.data.Dataset):
    def __init__(self, split='val', dir = '../data', 
                 pad_value = 0, pad_size = 64,
                 transform=None, target_transform=None):
        self.pad_value = pad_value
        self.pad_size = pad_size

        self.split = split
        self.split_dir = os.path.join(dir, 'boxes', split)

        self.transform = transform
        self.target_transform = target_transform

        self.boxes = []
        self.x_path = x_path = os.path.join(self.split_dir, 'images')
        if self.split != 'test':
            self.y_path = os.path.join(self.split_dir, 'labels')
        for file in os.listdir(x_path):
            patches = os.listdir(os.path.join(x_path, file))
            for patch in patches:
                boxes = os.listdir(os.path.join(x_path, file, patch))
                for box in boxes:
                    box_path = os.path.join(file, patch, box)
                    self.boxes.append(box_path)

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, idx):
        box = self.boxes[idx]

        x = np.load(os.path.join(self.x_path, box))
        x = torch.from_numpy(x)

        if self.transform: x = self.transform(x)
        
        if self.split == 'test':
            return x, -1 # return -1 as dummy label for test set
        
        y = np.load(os.path.join(self.y_path, box))
        y = torch.from_numpy(y)

        if self.target_transform: y = self.target_transform(y)
        
        x = pad_tensor(x, pad_value = self.pad_value, pad_size = self.pad_size)
        y = pad_tensor(y, pad_value = self.pad_value, pad_size = self.pad_size)

        return x, y