import os
from dataset import read_image
import numpy as np
import argparse
from tqdm import tqdm


def patch_volume(volume, patch_size, pad_value=0):
    """
    volume: np.array
    patch_size: 3-tuple
    """

    x, y, z = volume.shape
    px, py, pz = patch_size

    assert x % px == 0, "volume width must be divisible by patch size"
    assert y % py == 0, "volume height must be divisible by patch size"

    # pad z dimension to be divisible by patch size
    if z % pz != 0:
        volume = np.pad(volume, ((0, 0), (0, 0), (0, pz - z % pz)), 'constant', constant_values=pad_value)

    # split volume into patches
    patches = volume.reshape(x // px, px, y // py, py, -1, pz).swapaxes(1, 2).reshape(-1, px, py, pz)

    return patches


def parse_option(notebook=False):

    parser = argparse.ArgumentParser(description="RibFracPatcher")

    parser.add_argument('--split', type=str, default='val', help='train, val, or test')
    parser.add_argument('--data_dir', type=str, default='../data_dev', help='Path to data directory')
    parser.add_argument('--save_dir', type=str, default='../data_dev_patches', help='Path to save directory')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 128], help='Patch size')

    args = parser.parse_args() if not notebook else parser.parse_args(args=[])
    return args


def main(args):

    if not os.path.exists(os.path.join(args.save_dir, args.split)):
        os.makedirs(os.path.join(args.save_dir, args.split))
        os.makedirs(os.path.join(args.save_dir, args.split, 'images'))
        os.makedirs(os.path.join(args.save_dir, args.split, 'labels'))

    print('Splitting {} image data into patches...'.format(args.split))
    split_dir_images = os.path.join(args.data_dir, args.split, 'images')
    for file in tqdm(os.listdir(split_dir_images)):
        image_patches = patch_volume(read_image(os.path.join(split_dir_images, file))[0], args.patch_size)
        np.save(os.path.join(args.save_dir, args.split, 'images', file.split('.')[0]), image_patches)

    if args.split != 'test':
        split_dir_labels = os.path.join(args.data_dir, args.split, 'labels')
        print('Splitting {} label data into patches...'.format(args.split))
        for file in tqdm(os.listdir(split_dir_labels)):
            label_patches = patch_volume(read_image(os.path.join(split_dir_labels, file))[0], args.patch_size)
            np.save(os.path.join(args.save_dir, args.split, 'labels', file.split('.')[0]), label_patches)


if __name__ == '__main__':
    args = parse_option()
    print(args)
    main(args)
