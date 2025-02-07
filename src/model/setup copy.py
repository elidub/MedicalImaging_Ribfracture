import os
import torch.nn as nn
import torch

from src.model.modules import RetinaNetLoss
from src.model.learner import Learner, RetinanetLearner
from src.model.models import DummyNetwork, RetinaNet3D, UNet3D, FracNet

class BCEWithIgnoreLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super(BCEWithIgnoreLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # Ignore values marked with self.ignore_index
        valid_mask = target != self.ignore_index
        
        # Calculate the BCE loss for the valid values
        loss = nn.BCELoss()(input[valid_mask], target[valid_mask].float())
        
        return loss

def setup_model(args):
    net = args.net
    if net == "dummy":
        net = DummyNetwork()
        loss = nn.CrossEntropyLoss()
        learner = Learner
    elif net == "retinanet":
        net = RetinaNet3D(in_channels=1)
        loss = RetinaNetLoss()
        learner = RetinanetLearner
    elif net == "unet3d":
        net = UNet3D(1, 1)
        loss = BCEWithIgnoreLoss()
        learner = Learner
    elif net == 'fracnet':
        net = FracNet(1, 1)
        loss = BCEWithIgnoreLoss()
        learner = Learner
    else:
        raise ValueError("Invalid network type")

    if args.train:
        model = learner(net, loss)
    elif args.predict:
        ckpt_path = os.path.join(args.log_dir, args.net, args.version, "checkpoints")
        ckpt = os.listdir(ckpt_path)[0]
        assert len(os.listdir(ckpt_path)) == 1, "Multiple checkpoints found"
        model = learner.load_from_checkpoint(
            os.path.join(ckpt_path, ckpt), net=net, loss=loss,
            map_location=torch.device('cpu')
        )
        model = model.eval()

    return model
