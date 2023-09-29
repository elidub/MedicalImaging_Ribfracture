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

# From here it's about converting the segmentation mask to bounding boxes

def convert(path):
    """ Convert the segmentation mask to bounding boxes """ 

    annot, _ = read_image(path)
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


def new_annots(split_dir, split):
    """ Use each image per split to form a dictionary of bounding boxes """
    metadata = pd.read_csv(split_dir + f'/ribfrac-{split}-info.csv')

    annots = {}

    labels_dir = split_dir + '/labels'
    for file in os.listdir(labels_dir):

        im_id = file.split('-')[0]
        print(f'running for image: {im_id}')
        im_metadata = metadata[metadata['public_id'] == im_id]

        boxes, _ = convert(labels_dir + '/' + file)
        boxes.insert(0, [])

        # Convert all numpy int64 to int
        for i in range(len(boxes)):
            boxes[i] = [int(x) for x in boxes[i]]

        im_metadata['bbox'] = boxes
        annots[im_id] = im_metadata.loc[im_metadata['label_code'] != 0, 'bbox'].tolist()

    return annots

def process_split(split, data_dir, save_path=''):
    """ Process the train/val/test split and save the annotations as a json file """
    path = data_dir + split 
    anns = new_annots(path, split)

    with open(f"{split}.json", "w+") as outfile:
        json.dump(anns, outfile)

# Extract bounding boxes for a split
# process_split('val', "../../data_dev/")

# From here it's about extracting the image patches

def extract_gt_patches(img_path, annot_path, img_id):
    annot_data, _ = read_image(annot_path)
    img_data, _ = read_image(img_path)

    # Get the bounding boxes
    bounding_boxes, ids = convert(annot_path)

    img_patches = []
    annot_patches = []
    
    # Extract the patches
    for box in bounding_boxes:
        x, y, z, width, height, depth = box
        img_patch = img_data[x:x+width, y:y+height, z:z+depth]
        annot_patch = annot_data[x:x+width, y:y+height, z:z+depth]

        annot_patches.append(annot_patch)
        img_patches.append(img_patch)

    return img_patches, annot_patches 


def run_per_img(split, img_id):
    ''' Extract GT patches for the original images and annotations per split and image. '''

    img_path = f"../../data_dev/{split}/images/{img_id}-image.nii.gz"
    annot_path = f"../../data_dev/{split}/labels/{img_id}-label.nii.gz"

    img_patches, annot_patches = extract_gt_patches(img_path, annot_path, img_id)
    return img_patches, annot_patches 

def run_per_split(split):
    ''' Extract GT patches for the original images and annotations per split. 
    (for now I just save them in a dict, but we can save them as .nii.gz files)'''

    split_patches = {}

    path = f"../../data_dev/{split}/labels"
    for file in os.listdir(path):
        img_id = file.split('-')[0]
        
        print(f'running for image: {img_id}')
        img_patches, annot_patches = run_per_img(split, img_id)

        split_patches[img_id] = {'img_patches': img_patches, 'annot_patches': annot_patches}

    return split_patches                              

# Extract patches for a whole split
# split_patches = run_per_split('train')
