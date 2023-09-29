import torch
import pandas as pd
import os
import nibabel as nib
import gzip
from PIL import Image

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
    def __init__(self, split='val', dir = '../data_boxes', transform=None, target_transform=None):
        self.split = split
        self.split_dir = os.path.join(dir, split)

        self.transform = transform
        self.target_transform = target_transform

        self.IDs = []
        for filename in os.listdir(os.path.join(self.split_dir, 'images')):
            if filename.endswith(".pt"):
                self.IDs.append(filename)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        ID = self.IDs[idx]

        img_path = os.path.join(self.split_dir, 'images', ID)
        image = torch.load(img_path)
        if self.transform: image = self.transform(image)
        
        if self.split == 'test':
            return image, -1 # return -1 as dummy label for test set
    
        labels_path = os.path.join(self.split_dir, 'labels', ID)
        label = torch.load(labels_path)
        if self.target_transform: label = self.target_transform(label)

        return image, label