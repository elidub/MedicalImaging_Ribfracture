import torch
import torch.nn as nn

from torch.nn.functional import one_hot, cross_entropy
from src.model.anchors import Anchors3D, match_anchor_boxes


class BoxLabelEncoder:
    def __init__(
        self,
        num_channels=1,
        volume_width=64,
        volume_height=64,
        volume_depth=64,
        box_variance=[0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
    ):
        get_anchors = Anchors3D()
        sample = torch.zeros(
            (1, num_channels, volume_width, volume_height, volume_depth)
        )

        # self.dim_norm = torch.tensor(
        #     [
        #         volume_width,
        #         volume_height,
        #         volume_depth,
        #         volume_width,
        #         volume_height,
        #         volume_depth,
        #     ]
        # )

        self.anchors = get_anchors(sample)
        self.box_variance = torch.tensor(box_variance)

    def encode(self, gt_box, gt_cls):
        b = gt_box.shape[0]
        box_res = torch.zeros((b, self.anchors.shape[0], 6))
        cls_res = torch.zeros((b, self.anchors.shape[0], 1))

        # Fix encoding for binary classification
        if gt_cls.max() < 2:
            gt_cls -= 1

        for i in range(b):
            # Match anchors and gt boxes to create targets
            match_idx, pos_mask, neg_mask = match_anchor_boxes(self.anchors, gt_box[i])
            match_box = torch.index_select(gt_box[i], 0, match_idx)

            box_res[i] = (
                torch.cat(
                    [
                        (match_box[:, :3] - self.anchors[:, :3]) / self.anchors[:, 3:],
                        torch.log(match_box[:, 3:]) / self.anchors[:, 3:],
                    ],
                    dim=-1,
                )
                # / self.dim_norm
                / self.box_variance
            )

            # Create classification targets
            match_cls = torch.index_select(gt_cls[i], 0, match_idx)
            match_cls = torch.where(pos_mask.unsqueeze(-1) != 1.0, -1.0, match_cls)
            cls_res[i] = torch.where(neg_mask.unsqueeze(-1) == 1.0, -2.0, match_cls)
        return box_res, cls_res.squeeze(-1).long() + 2


class BoxLabelDecoder:
    def __init__(
        self,
        num_channels=1,
        volume_width=64,
        volume_height=64,
        volume_depth=64,
        box_variance=[0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
    ):
        get_anchors = Anchors3D()
        sample = torch.zeros(
            (1, num_channels, volume_width, volume_height, volume_depth)
        )

        # self.dim_norm = torch.tensor(
        #     [
        #         volume_width,
        #         volume_height,
        #         volume_depth,
        #         volume_width,
        #         volume_height,
        #         volume_depth,
        #     ]
        # )

        self.anchors = get_anchors(sample)
        self.box_variance = torch.tensor(box_variance)

    def decode(self, pred_box, pred_cls):
        b = pred_box.shape[0]
        box_res = torch.zeros((b, self.anchors.shape[0], 6))
        cls_res = torch.zeros((b, self.anchors.shape[0], pred_cls.shape[-1]))

        pred_box *= self.box_variance
        # pred_box *= self.dim_norm

        for i in range(b):
            box_res[i] = torch.cat(
                [
                    pred_box[i, :, :3] * self.anchors[:, 3:] + self.anchors[:, :3],
                    torch.exp(pred_box[i, :, 3:] * self.anchors[:, 3:]),
                ],
                dim=-1,
            )

            cls_res[i] = pred_cls[i].softmax(dim=-1)

        return box_res, cls_res


class RetinaNetLoss(nn.Module):
    def __init__(
        self,
        num_classes=1,
        alpha=0.25,
        gamma=2.0,
        delta=1.0,
    ):
        super().__init__()

        self.num_classes = num_classes + 2
        self.alpha = 0.25
        self.gamma = 2.0
        self.delta = 1.0

    def forward(self, pred_box, pred_cls, gt_box, gt_cls):
        device = pred_box.device
        pred_box.shape[0]

        # Extract one-hot labels and masks
        gt_cls_oh = one_hot(gt_cls.long(), self.num_classes).float()
        pos_mask = (gt_cls > 1).float().squeeze(-1)
        ign_mask = (gt_cls == 0).float().squeeze(-1)

        # Compute L1 loss between gt boxes and predictions
        box_loss_abs = (pred_box - gt_box).abs()
        box_loss_sqr = box_loss_abs**2
        box_loss = torch.where(
            box_loss_abs < self.delta, 0.5 * box_loss_sqr, box_loss_abs - 0.5
        ).sum(dim=-1)

        # Compute focal loss between gt classes and predictions
        p = pred_cls.sigmoid()
        ce = (
            cross_entropy(pred_cls.permute(0, 2, 1), gt_cls, reduce=False)
            .unsqueeze(-1)
            .repeat(1, 1, self.num_classes)
        )

        alpha = torch.where(gt_cls_oh == 1.0, self.alpha, (1.0 - self.alpha))
        pt = torch.where(gt_cls_oh == 1.0, p, 1 - p)
        cls_loss = (alpha * (1.0 - pt) ** self.gamma * ce).sum(dim=-1)

        # Apply masks to L1 loss and focal loss
        box_loss = torch.where(pos_mask == 1.0, box_loss, 0.0)
        cls_loss = torch.where(ign_mask == 1.0, 0.0, cls_loss)

        pos_mask_norm = pos_mask.sum(dim=-1) + 1e-7
        box_loss = box_loss.sum(dim=-1) / pos_mask_norm
        cls_loss = cls_loss.sum(dim=-1) / pos_mask_norm

        return box_loss.sum(dim=-1), cls_loss.sum(dim=-1)
