import numpy as np


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

