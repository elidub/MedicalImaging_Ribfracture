import os
import sys
import gzip
import torch
import numpy as np
import pandas as pd
import nibabel as nib

from PIL import Image
from tqdm import tqdm

sys.path.insert(1, sys.path[0] + "/..")
from src.data.utils import pad_tensor
from src.model.modules import BoxLabelEncoder
from src.data.utils import simplify_labels


def read_image(file_path):
    nii_img = nib.load(file_path)
    image_data = nii_img.get_fdata()
    header = nii_img.header
    return image_data, header


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(
        self, split="val", dir="../data_dev", transform=None, target_transform=None
    ):
        self.split = split
        self.split_dir = os.path.join(dir, "raw", split)

        self.transform = transform
        self.target_transform = target_transform

        self.IDs = []
        for filename in os.listdir(os.path.join(self.split_dir, "images")):
            if filename.endswith(".nii.gz"):
                self.IDs.append(filename.split("-")[0])

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        ID = self.IDs[idx]

        img_path = os.path.join(self.split_dir, "images", f"{ID}-image.nii.gz")
        image, _ = read_image(img_path)
        if self.transform:
            image = self.transform(image)

        if self.split == "test":
            return image, -1  # return -1 as dummy label for test set

        labels_path = os.path.join(self.split_dir, "labels", f"{ID}-label.nii.gz")
        label, _ = read_image(labels_path)
        if self.target_transform:
            label = self.target_transform(label)

        return image, simplify_labels(label)


class BoxesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split="val",
        dir="../data",
        pad_value=0,
        pad_size=64,
        transform=None,
        target_transform=None,
    ):
        self.pad_value = pad_value
        self.pad_size = pad_size

        self.split = split
        self.split_dir = os.path.join(dir, "boxes", split)

        self.transform = transform
        self.target_transform = target_transform

        self.boxes = []
        self.x_path = x_path = os.path.join(self.split_dir, "images")
        if self.split != "test":
            self.y_path = os.path.join(self.split_dir, "labels")
        for file in os.listdir(x_path):
            patches = os.listdir(os.path.join(x_path, file))
            # patches = patches[:2] # SELECT ONLY FIRST TWO PATCHES
            patches = [patch for patch in patches if patch != "metadata.csv"]
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
        x = x[list(x.keys())[0]]
        x = torch.from_numpy(x)

        if self.transform:
            x = self.transform(x)

        x = pad_tensor(x, pad_value=self.pad_value, pad_size=self.pad_size)
        if self.split == "test":
            return x, torch.tensor([-1])  # return -1 as dummy label for test set

        y = np.load(os.path.join(self.y_path, box))
        y = y[list(y.keys())[0]]
        y = torch.from_numpy(y)

        if self.target_transform:
            y = self.target_transform(y)

        y = simplify_labels(y) 
        y = pad_tensor(y, pad_value=-1, pad_size=self.pad_size)

        return x, y


class PatchesDataset(torch.utils.data.Dataset):
    def __init__(
        self, split="train", dir="data", transform=None, target_transform=None
    ):
        self.split = split
        self.img_dir = os.path.join(dir, "patches", split, "images")
        self.lab_dir = os.path.join(dir, "boxes", split, "images")

        self.transform = transform
        self.target_transform = target_transform

        self.idx_to_im = []
        self.idx_to_file = []
        self.patches_per_im = []

        print("Initializing patches dataset...")

        if not os.path.exists(f"idx_to_im_{split}.npy"):
            for file in tqdm(os.listdir(self.img_dir)):
                if not file.endswith(".npz"):
                    continue

                im_data = np.load(os.path.join(self.img_dir, file))["arr_0"]
                num_patches = im_data.shape[0]

                self.idx_to_im.append(sum(self.patches_per_im))
                self.patches_per_im.append(num_patches)
                self.idx_to_file.append(file.replace(".npz", ""))

            np.save(f"idx_to_im_{split}.npy", self.idx_to_im)
            np.save(f"idx_to_file_{split}.npy", self.idx_to_file)
            np.save(f"patches_per_im_{split}.npy", self.patches_per_im)
        else:
            self.idx_to_im = np.load(f"idx_to_im_{split}.npy")
            self.idx_to_file = np.load(f"idx_to_file_{split}.npy")
            self.patches_per_im = np.load(f"patches_per_im_{split}.npy")

        self.idx_to_im = torch.tensor(self.idx_to_im)
        self.label_encoder = BoxLabelEncoder(
            volume_width=128, volume_depth=128, volume_height=128
        )

    def __len__(self):
        return sum(self.patches_per_im)

    def __getitem__(self, idx):
        lookup = self.idx_to_im - idx
        lookup[lookup > 0] = 1000

        img_idx = torch.argmin(lookup.abs()).item()
        pat_idx = idx - self.idx_to_im[img_idx].item()

        file = f"{self.idx_to_file[img_idx]}.npz"
        img = np.load(os.path.join(self.img_dir, file))["arr_0"][pat_idx]
        img = torch.tensor(img).unsqueeze(0)

        img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=2).clamp(
            min=0, max=1
        )  # Upsample

        boxes = torch.zeros((1, 0, 6))
        classes = torch.zeros((1, 0))
        has_boxes = False

        if self.split != "test":
            if os.path.exists(
                os.path.join(self.lab_dir, self.idx_to_file[img_idx], "metadata.csv")
            ):
                label_data = pd.read_csv(
                    os.path.join(
                        self.lab_dir, self.idx_to_file[img_idx], "metadata.csv"
                    )
                )

                boxes = label_data[label_data["patch_id"].astype(int) == pat_idx]

                if len(boxes) != 0:
                    boxes = torch.tensor(
                        boxes[["x", "y", "z", "width", "height", "depth"]].to_numpy()
                    ).unsqueeze(0)

                    boxes *= 2  # Usample

                    classes = torch.ones((1, boxes.shape[1], 1))
                    boxes, classes = self.label_encoder.encode(boxes, classes)
                    has_boxes = True
                else:
                    boxes = torch.zeros((1, 0, 6))
                    classes = torch.zeros((1, 0))

        return (
            img.squeeze(0),
            boxes.squeeze(0),
            classes.squeeze(0),
            {
                "file": self.idx_to_file[img_idx],
                "patch": pat_idx,
                "has_boxes": has_boxes,
            },
        )
