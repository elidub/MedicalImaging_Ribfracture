"""
Mostly copy-paste from https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/anchors.py

Few modifications made for handling 3D anchors
"""

import torch
import torch.nn as nn
import numpy as np


class Anchors3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.pyramid_levels = [3, 4, 5, 6, 7]
        self.strides = [2**e for e in self.pyramid_levels]
        self.sizes = [2 ** (e + 2) for e in self.pyramid_levels]

        self.ratios = np.array([0.5, 1, 2])
        self.scales = np.array([2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, x):
        x_shape = np.array(x.shape[2:])
        x_shapes = [(x_shape + 2**e - 1) // (2**e) for e in self.pyramid_levels]

        # Compute anchors over all pyramid levels in 3D (hence x, y, z, w, h, d)
        all_anchors = np.zeros((0, 6)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(
                base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales
            )

            shifted_anchors = shift(x_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)
        return torch.from_numpy(all_anchors.astype(np.float32)).to(x.device)


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 6))

    # scale base_size
    anchors[:, 3:] = base_size * np.tile(scales, (3, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 3] * anchors[:, 4] * anchors[:, 5]

    # correct for ratios
    anchors[:, 3] = (areas / np.repeat(ratios, len(scales))) ** (1 / 3)
    anchors[:, 4] = anchors[:, 3] * np.repeat(ratios, len(scales))
    anchors[:, 5] = anchors[:, 3] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, z_ctr, w, h, d) -> (x1, y1, z1, x2, y2, z2)
    anchors[:, 0::3] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    anchors[:, 1::3] -= np.tile(anchors[:, 4] * 0.5, (2, 1)).T
    anchors[:, 2::3] -= np.tile(anchors[:, 5] * 0.5, (2, 1)).T

    return anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[2]) + 0.5) * stride
    shift_y = (np.arange(0, shape[1]) + 0.5) * stride
    shift_z = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y, shift_z = np.meshgrid(shift_x, shift_y, shift_z)

    shifts = np.vstack(
        (
            shift_x.ravel(),
            shift_y.ravel(),
            shift_z.ravel(),
            shift_x.ravel(),
            shift_y.ravel(),
            shift_z.ravel(),
        )
    ).transpose()

    # add A anchors (1, A, 6) to
    # cell K shifts (K, 1, 6) to get
    # shift anchors (K, A, 6)
    # reshape to (K*A, 6) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = anchors.reshape((1, A, 6)) + shifts.reshape((1, K, 6)).transpose(
        (1, 0, 2)
    )

    all_anchors = all_anchors.reshape((K * A, 6))

    return all_anchors
