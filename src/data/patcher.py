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
        z = volume.shape[2]

    # split volume into patches
    patches = []
    for i in range(x // px):
        for j in range(y // py):
            for k in range(z // pz):
                patches.append(volume[i * px:(i + 1) * px, j * py:(j + 1) * py, k * pz:(k + 1) * pz])

    return np.array(patches)


def reconstruct_volume(patches, original_shape, pad_value=0):
    """
    patches: np.array of shape (num_patches, px, py, pz)
    original_shape: 3-tuple representing the shape of the original volume
    patch_size: 3-tuple representing the patch size used in patch_volume function
    pad_value: value used for padding in patch_volume function (default is 0)
    """
    
    num_patches, px, py, pz = patches.shape
    x, y, z = original_shape
    z = z if z % pz == 0 else z + pz - z % pz  # pad z dimension to be divisible by patch size
    reconstructed_volume = np.full((x, y, z), fill_value=pad_value)

    assert x % px == 0, "volume width must be divisible by patch size"
    assert y % py == 0, "volume height must be divisible by patch size"
    assert z % pz == 0, "volume depth must be divisible by patch size"
    assert num_patches == (x // px) * (y // py) * (z // pz), "Invalid number of patches for given volume shape and patch size"

    for i in range(num_patches):
        x_start = (i // ((y // py) * (z // pz))) * px
        y_start = ((i // (z // pz)) % (y // py)) * py
        z_start = (i % (z // pz)) * pz
        reconstructed_volume[x_start:x_start + px, y_start:y_start + py, z_start:z_start + pz] = patches[i]

    return reconstructed_volume[:, :, :original_shape[2]]

