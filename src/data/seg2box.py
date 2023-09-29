import numpy as np
import pandas as pd
import os
import nibabel as nib
import json
import gzip
from scipy.ndimage import label

def read_image(file_path):
    nii_img = nib.load(file_path)
    image_data = nii_img.get_fdata()
    header = nii_img.header
    return image_data, header


def convert(split_dir, ID):
    """ Convert the segmentation mask to bounding boxes """ 
    labels_path = os.path.join(split_dir, 'labels', f"{ID}-label.nii.gz")

    annot, _ = read_image(labels_path)
    num_segments = np.unique(annot)

    bounding_boxes = []
    ids = []

    for segment_label in num_segments[1:]:
        
        # boolean mask for the current segment
        segment_mask = annot == segment_label
        
        min_coords = np.min(np.where(segment_mask), axis=1)
        max_coords = np.max(np.where(segment_mask), axis=1)
        
        width = max_coords[0] - min_coords[0] + 1
        height = max_coords[1] - min_coords[1] + 1
        depth = max_coords[2] - min_coords[2] + 1
        
        # Calculate the (x, y, z) coordinates of the bounding box
        x = min_coords[0]
        y = min_coords[1]
        z = min_coords[2]

        bounding_box = [x, y, z, width, height, depth]
        bounding_boxes.append(bounding_box)

        ids.append(segment_label)

    return bounding_boxes, ids


def new_annots(split_dir):
    """ Use each image per split to form a dictionary of bounding boxes """
    metadata = pd.read_csv(split_dir + f'/ribfrac-{split_dir}-info.csv')

    annots = {}

    for file in os.listdir(split_dir + '/labels'):
        im_id = file.split('-')[0]
        im_metadata = metadata[metadata['public_id'] == im_id]

        boxes, ids = convert(split_dir, im_id)
        boxes.insert(0, [])

        # Convert all numpy int64 to int
        for i in range(len(boxes)):
            boxes[i] = [int(x) for x in boxes[i]]

        im_metadata['bbox'] = boxes
        annots[im_id] = im_metadata.loc[im_metadata['label_code'] != 0, 'bbox'].tolist()

    return annots

def process_split(split, path=''):
    """ Process the train/val/test split and save the annotations as a json file """
    anns = new_annots(split)

    with open(f"{path}{split}.json", "w+") as outfile:
        json.dump(anns, outfile)

process_split('train')
