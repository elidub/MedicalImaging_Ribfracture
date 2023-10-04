"""
Mostly copy-paste from https://keras.io/examples/vision/retinanet/

Few modifications made for handling 3D anchors
"""

import torch
import torch.nn as nn

from src.misc.utils import pairwise_iou, bbox_centerwhd_to_xyzwhd


class Anchors3D(nn.Module):
    """Generates 3D anchor boxes.

    This class has operations to generate 3D anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor box is of the format `[x, y, z, width, height, depth]`.

    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      volumes: A list of float values representing the volumes of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self):
        super().__init__()

        self.ratios = [
            (0.5, 1.0, 2.0),
            (1.0, 1.0, 1.0),
            (0.5, 1.0, 1.5),
            (1.0, 0.75, 1.5),
            (1.5, 1.0, 0.5),
        ]

        self.scales = [2**e for e in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.ratios) * len(self.scales)
        self._strides = [2**i for i in range(2, 7)]
        # self._volumes = [e**3 for e in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._volumes = [e**3 for e in [4.0, 8.0, 16.0, 32.0, 64.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for volume in self._volumes:
            anchor_dims = []

            for ratio in self.ratios:
                anchor_width = (volume / (ratio[1] * ratio[2])) ** (1.0 / 3.0)
                anchor_height = (volume / (ratio[0] * ratio[2])) ** (1.0 / 3.0)
                anchor_depth = (volume / (ratio[0] * ratio[1])) ** (1.0 / 3.0)

                dims = torch.tensor([anchor_width, anchor_height, anchor_depth]).view(
                    1, 1, 3
                )

                for scale in self.scales:
                    anchor_dims.append(scale * dims)

            anchor_dims_all.append(torch.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, feature_depth, level):
        """Generates anchor boxes for a given 3D feature map size and level

        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          feature_depth: An integer representing the depth of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.

        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * feature_depth * num_anchors, 6)`
        """
        rx = torch.tensor(range(0, feature_width)) + 0.5
        ry = torch.tensor(range(0, feature_height)) + 0.5
        rz = torch.tensor(range(0, feature_depth)) + 0.5

        stride = self._strides[level - 2]

        centers = torch.stack(torch.meshgrid(rx, ry, rz), axis=-1) * stride
        centers = centers.unsqueeze(-2)
        centers = torch.tile(centers, [1, 1, 1, self._num_anchors, 1])

        dims = torch.tile(
            self._anchor_dims[level - 2],
            [feature_height, feature_width, feature_depth, 1, 1],
        )

        anchors = torch.cat([centers, dims], axis=-1)

        return anchors.reshape(
            feature_height * feature_width * feature_depth * self._num_anchors, 6
        )

    def forward(self, x):
        """Generates 3D anchor boxes for all the feature maps of the feature pyramid.

        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.
          image_depth: Depth of the input image.

        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 6)`
        """
        image_height, image_width, image_depth = x.shape[2:]

        anchors = [
            self._get_anchors(
                torch.math.ceil(image_height / 2**i),
                torch.math.ceil(image_width / 2**i),
                torch.math.ceil(image_depth / 2**i),
                i,
            )
            for i in range(2, 7)
        ]

        return bbox_centerwhd_to_xyzwhd(torch.cat(anchors, axis=0))


def match_anchor_boxes(anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4):
# def match_anchor_boxes(anchor_boxes, gt_boxes, match_iou=0.001, ignore_iou=0.0001):
    iou_matrix = pairwise_iou(anchor_boxes, gt_boxes)

    max_iou, matched_gt_idx = torch.max(iou_matrix, dim=1, keepdim=False)

    positive_mask = max_iou >= match_iou
    negative_mask = max_iou < ignore_iou

    ignore_mask = torch.logical_not(torch.logical_or(positive_mask, negative_mask))

    return (matched_gt_idx, positive_mask.float(), ignore_mask.float())
