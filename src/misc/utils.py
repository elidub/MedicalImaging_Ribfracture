import lightning.pytorch as pl
import torch


def set_seed_and_precision(args):
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")


def bbox_xyzwhd_to_corners(boxes):
    # Extract individual components from the batched tensor
    x, y, z, w, h, d = torch.split(boxes, 1, dim=-1)

    # Calculate the coordinates of the two opposite corners for each box in the batch
    x1 = x - w / 2
    y1 = y - h / 2
    z1 = z - d / 2
    x2 = x + w / 2
    y2 = y + h / 2
    z2 = z + d / 2

    # Stack the results into a new tensor of shape (batch_size, 6)
    return torch.cat([x1, y1, z1, x2, y2, z2], dim=-1)


def bbox_centerwhd_to_xyzwhd(boxes):
    cx, cy, cz, w, h, d = torch.split(boxes, 1, dim=-1)

    # Calculate top-left coordinates
    x = cx - w / 2
    y = cy - h / 2
    z = cz - d / 2

    # Stack the results into a new tensor of shape (batch_size, 6)
    return torch.cat([x, y, x, w, h, d], dim=-1)


def pairwise_iou(boxes1, boxes2):
    boxes1 = bbox_xyzwhd_to_corners(boxes1)
    boxes2 = bbox_xyzwhd_to_corners(boxes2)

    # Reshape the input boxes for broadcasting
    boxes1 = boxes1.view(-1, 1, 6)  # [M, 1, 6]
    boxes2 = boxes2.view(1, -1, 6)  # [1, N, 6]

    # Calculate the intersection volume
    min_coords = torch.max(boxes1[..., :3], boxes2[..., :3])
    max_coords = torch.min(boxes1[..., 3:], boxes2[..., 3:])
    intersection_dims = torch.clamp(max_coords - min_coords, min=0)
    intersection_volume = torch.prod(intersection_dims, dim=2)  # [M, N]

    # Calculate the volume of each box
    volume1 = torch.prod(boxes1[..., 3:] - boxes1[..., :3], dim=2)  # [M, 1]
    volume2 = torch.prod(boxes2[..., 3:] - boxes2[..., :3], dim=2)  # [1, N]

    # Calculate the union volume
    union_volume = volume1 + volume2 - intersection_volume  # [M, N]

    # Calculate IoU
    return intersection_volume / union_volume
