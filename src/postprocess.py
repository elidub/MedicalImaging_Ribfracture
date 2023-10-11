import os

import argparse
import csv
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects
from tqdm import tqdm

from data.dataset import read_image
from data.seg2box import stich_boxes_to_patch
from data.patcher import reconstruct_volume


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
    parser.add_argument('--save_dir', type=str, default='../logs/unet3d/version1/submissions', help='Path to data directory')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 128], help='Patch size')

    args = parser.parse_args() if not notebook else parser.parse_args(args=[])
    return args


def main(args):

    pred_info_list = []

    for img_id in tqdm(os.listdir(os.path.join(args.prediction_box_dir, args.split))):
        original_image, _ = read_image(
            os.path.join(args.original_image_dir, 'raw', args.split, 'images', f'{img_id}-image.nii.gz')
            )
        with open(os.path.join(args.prediction_box_dir, args.split, img_id, 'metadata.csv'), mode='r') as file:
            csv_reader = csv.reader(file)
            bounding_boxes = {rows[0]: rows[1] for rows in csv_reader}
        print(bounding_boxes)
        for key in bounding_boxes.keys():
            predictions = []
            for i, box in enumerate(os.listdir(os.path.join(args.prediction_box_dir, args.split, img_id, f'patch{key}'))):
                prediction = np.load(os.path.join(args.box_dir, args.split, img_id, f'patch{key}', box))
                prediction = prediction[:bounding_boxes[key][i][3], :bounding_boxes[key][i][4], :bounding_boxes[key][i][5]]
                predictions.append(prediction)
            patched = stich_boxes_to_patch(bounding_boxes[key], predictions, args.patch_size)
        pred_arr = reconstruct_volume(patched, original_image.shape)
        pred_arr = _post_process(pred_arr, original_image, prob_thresh=0.5, bone_thresh=200, size_thresh=1000)

        pred_image, pred_info = _make_submission_files(pred_arr, img_id, np.eye(4)) # TODO: check/add affine (np.eye(4))
        pred_info_list.append(pred_info)
        pred_path = os.path.join(args.save_dir, f"{img_id}_pred.nii.gz")
        nib.save(pred_image, pred_path)


if __name__ == '__main__':
    args = parse_option()
    print(args)
    main(args)

