import os, sys
import argparse
import csv
import pandas as pd
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects
from tqdm import tqdm

sys.path.insert(1, sys.path[0] + "/..")
from src.data.dataset import read_image
from src.data.seg2box import stich_boxes_to_patch
from src.data.patcher import patch_volume, reconstruct_volume


def read_metadata(path):
    '''
        Reads metadata from csv file and returns the bounding boxes for each image.
        Metadata is structured as patch_id, box_id, x, y, z, width, height, depth.
        Bounding boxes are returned as a dictionary of lists of lists with keys as patch_id and outer list as box_id.
        Inner list contains x, y, z, width, height, depth.
    '''

    bounding_boxes = {}
    with open(path, mode='r') as file:
        csv_reader = csv.reader(file)
        for i, row in enumerate(csv_reader):
            if i > 0:
                if row[0] not in bounding_boxes.keys():
                    bounding_boxes[row[0]] = [row[2:]]
                else:
                    bounding_boxes[row[0]].append(row[2:])
    return bounding_boxes


def _remove_low_probs(pred, prob_thresh):
    pred = np.where(pred > prob_thresh, pred, 0)

    return pred


def _remove_spine_fp(pred, image, bone_thresh):
    image_bone = image > bone_thresh
    image_bone_2d = image_bone.sum(axis=-1)
    image_bone_2d = ndimage.median_filter(image_bone_2d, 10)
    image_spine = (image_bone_2d > image_bone_2d.max() // 3)
    kernel = disk(7)
    image_spine = ndimage.binary_opening(image_spine, kernel)
    image_spine = ndimage.binary_closing(image_spine, kernel)
    image_spine_label = label(image_spine)
    max_area = 0

    for region in regionprops(image_spine_label):
        if region.area > max_area:
            max_region = region
            max_area = max_region.area
    image_spine = np.zeros_like(image_spine)
    image_spine[
        max_region.bbox[0]:max_region.bbox[2],
        max_region.bbox[1]:max_region.bbox[3]
    ] = max_region.convex_image > 0

    return np.where(image_spine[..., np.newaxis], 0, pred)


def _remove_small_objects(pred, size_thresh):
    pred_bin = pred > 0
    pred_bin = remove_small_objects(pred_bin, size_thresh)
    pred = np.where(pred_bin, pred, 0)

    return pred


def _post_process(pred, image, prob_thresh, bone_thresh, size_thresh):

    # remove connected regions with low confidence
    pred = _remove_low_probs(pred, prob_thresh)

    # remove spine false positives
    pred = _remove_spine_fp(pred, image, bone_thresh)

    # remove small connected regions
    pred = _remove_small_objects(pred, size_thresh)

    return pred


def _make_submission_files(pred, image_id, affine):
    pred_label = label(pred > 0).astype(np.int16)
    pred_regions = regionprops(pred_label, pred)
    pred_index = [0] + [region.label for region in pred_regions]
    pred_proba = [0.0] + [region.mean_intensity for region in pred_regions]
    # placeholder for label class since classifaction isn't included
    pred_label_code = [0] + [1] * int(pred_label.max())
    pred_image = nib.Nifti1Image(pred_label, affine)
    pred_info = pd.DataFrame({
        "public_id": [image_id] * len(pred_index),
        "label_id": pred_index,
        "confidence": pred_proba,
        "label_code": pred_label_code
    })

    return pred_image, pred_info


def parse_option(notebook=False):

    parser = argparse.ArgumentParser(description="RibFracPatcher")

    parser.add_argument('--split', type=str, default='val', help='train, val, or test')
    parser.add_argument(
        '--prediction_box_dir',
        type=str,
        default='../logs/unet3d/version_1/segmentations',
        help='Path to prediction data directory'
        )
    parser.add_argument(
        '--original_image_dir',
        type=str,
        default='../data',
        help='Path to original image data directory'
        )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='../logs/submissions/version_1',
        help='Path to data directory')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 128], help='Patch size')
    parser.add_argument('--prob_thresh', type=float, default=0.5, help='Probability threshold')
    parser.add_argument('--bone_thresh', type=float, default=200, help='Bone threshold')
    parser.add_argument('--size_thresh', type=int, default=1000, help='Size threshold')

    args = parser.parse_args() if not notebook else parser.parse_args(args=[])
    return args


def main(args):

    pred_info_list = []

    for img_id in tqdm(os.listdir(os.path.join(args.prediction_box_dir, args.split))):
        
        original_image, _ = read_image(
            os.path.join(args.original_image_dir, 'raw', args.split, 'images', f'{img_id}-image.nii.gz')
            )
        bounding_boxes = read_metadata(
            os.path.join(args.prediction_box_dir, args.split, img_id, 'metadata.csv')
            )
        patches = patch_volume(np.zeros_like(original_image), args.patch_size, pad_value=0)
        
        for key in bounding_boxes.keys():
            predictions = []

            for i, box in enumerate(os.listdir(os.path.join(args.prediction_box_dir, args.split, img_id, f'patch{key}'))):
                prediction = np.load(os.path.join(args.prediction_box_dir, args.split, img_id, f'patch{key}', box))
                prediction = prediction[list(prediction.keys())[0]]
                w, h, d = int(bounding_boxes[key][i][3]), int(bounding_boxes[key][i][4]), int(bounding_boxes[key][i][5])
                prediction = prediction[:min(w, prediction.shape[0]), :min(h, prediction.shape[1]), :min(d, prediction.shape[2])]
                predictions.append(prediction)

            patched = stich_boxes_to_patch(bounding_boxes[key], predictions, args.patch_size)
            # TODO: think about overlapping boxes (currently overwriting with last prediction, could be average)
            patches[int(key)] = patched

        pred_arr = reconstruct_volume(patches, original_image.shape)

        # return pred_arr

        pred_arr = _post_process(pred_arr, original_image, args.prob_thresh, args.bone_thresh, args.size_thresh)
        pred_image, pred_info = _make_submission_files(pred_arr, img_id, np.eye(4)) # TODO: check/add affine (np.eye(4))
        pred_info_list.append(pred_info)
        pred_path = os.path.join(args.save_dir, f"{img_id}_pred.nii.gz")
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        nib.save(pred_image, pred_path)


if __name__ == '__main__':
    args = parse_option()
    print(args)
    main(args)

