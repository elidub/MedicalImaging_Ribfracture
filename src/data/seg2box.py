import numpy as np
import pandas as pd
import os
import nibabel as nib
import json
import gzip
from scipy.ndimage import label


def convert(annot):
    """ Convert the segmentation mask to bounding boxes """ 

    num_segments = np.unique(annot)
    bounding_boxes = []

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

    return bounding_boxes


def extract_boxes(img_data, annot_data):

    # Get the bounding boxes
    bounding_boxes = convert(annot_data)

    img_box = []
    annot_box = []
    
    # Extract the patches
    for box in bounding_boxes:
        x, y, z, width, height, depth = box
        img_patch = img_data[x:x+width, y:y+height, z:z+depth]
        annot_patch = annot_data[x:x+width, y:y+height, z:z+depth]

        annot_box.append(annot_patch)
        img_box.append(img_patch)

    return img_box, annot_box, bounding_boxes


def extract_boxes_from_patches(img_patches, annot_patches):

    img_boxes = []
    annot_boxes = []
    bounding_boxes_in_patch = {}

    for i in range(len(img_patches)):
        img_box, annot_box, bounding_boxes = extract_boxes(img_patches[i], annot_patches[i])
        img_boxes.append(img_box)
        annot_boxes.append(annot_box)
        bounding_boxes_in_patch[i] = bounding_boxes

    return img_boxes, annot_boxes, bounding_boxes_in_patch


def stich_boxes_to_patch(bounding_boxes, predictions, patch_shape):
    """
    Stitch the segmentations back to the original image level instead of bounding box prediction level.
    
    args:
        bounding_boxes: list of bounding box coordinates of shape (box_id, x, y, z, width, height, depth),
                        empty list if no bounding box in patch
        predictions: list of predictions of shape (box_id, width, height, depth)
        patch_shape: shape of the patch
    """

    # Create an empty array for the predictions
    patch_predictions = np.zeros(patch_shape)

    # Stitch the predictions back to the original image level
    for box, prediction in zip(bounding_boxes, predictions):
        x, y, z, width, height, depth = box
        x, y, z, width, height, depth = int(x), int(y), int(z), int(width), int(height), int(depth)
        patch_predictions[x:x+width, y:y+height, z:z+depth] = prediction

    return patch_predictions

