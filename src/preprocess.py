import os
import argparse
import numpy as np
from tqdm import tqdm
from data.dataset import read_image
from data.patcher import patch_volume
from data.seg2box import extract_boxes_from_patches
from data.utils import normalize, simplify_labels, extrapolate_bones


def parse_option(notebook=False):

    parser = argparse.ArgumentParser(description="RibFracPatcher")

    parser.add_argument('--split', type=str, default='val', help='train, val, or test')
    parser.add_argument('--data_dir', type=str, default='../data_dev', help='Path to data directory')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 128], help='Patch size')

    args = parser.parse_args() if not notebook else parser.parse_args(args=[])
    return args


def main(args):

    patch_dir = os.path.join(args.data_dir, 'patches')
    if not os.path.exists(os.path.join(patch_dir, args.split)):
        os.makedirs(os.path.join(patch_dir, args.split))
        os.makedirs(os.path.join(patch_dir, args.split, 'images'))
        if args.split != 'test':
            os.makedirs(os.path.join(patch_dir, args.split, 'labels'))

    box_dir = os.path.join(args.data_dir, 'boxes')
    if not os.path.exists(os.path.join(box_dir, args.split)):
        os.makedirs(os.path.join(box_dir, args.split))
        os.makedirs(os.path.join(box_dir, args.split, 'images'))
        if args.split != 'test':
            os.makedirs(os.path.join(box_dir, args.split, 'labels'))

    print('Splitting {} data into patches and creating bounding boxes...'.format(args.split))
    split_dir_images = os.path.join(args.data_dir, 'raw', args.split, 'images')
    for file in tqdm(os.listdir(split_dir_images)):
        img_id = file.split('-')[0]
        img_path = f"{args.data_dir}/raw/{args.split}/images/{img_id}-image.nii.gz"
        label_path = f"{args.data_dir}/raw/{args.split}/labels/{img_id}-label.nii.gz"
        img_data, _ = read_image(img_path)
        img_data = extrapolate_bones(img_data)
        img_data = normalize(img_data)
        img_patches = patch_volume(img_data, args.patch_size)
        np.save(os.path.join(patch_dir, args.split, 'images', f'{img_id}-image.npy'), img_patches)
        if args.split != 'test':
            label_data, _ = read_image(label_path)
            label_patches = patch_volume(label_data, args.patch_size)
            np.save(os.path.join(patch_dir, args.split, 'labels', f'{img_id}-label.npy'), label_patches)
            img_boxes, label_boxes = extract_boxes_from_patches(img_patches, label_patches)
            np.save(os.path.join(box_dir, args.split, 'images', f'{img_id}-image.npy'), img_boxes)
            np.save(os.path.join(box_dir, args.split, 'labels', f'{img_id}-label.npy'), label_boxes)


if __name__ == '__main__':
    args = parse_option()
    print(args)
    main(args)
