import os
import torch.nn as nn
import torch

from src.model.modules import RetinaNetLoss
from src.model.learner import Learner, RetinanetLearner
from src.model.models import DummyNetwork, RetinaNet3D, UNet3D


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
        loss = nn.BCELoss()
        learner = Learner
    else:
        raise ValueError("Invalid network type")

    if args.train:
        model = learner(net, loss)
    elif args.predict:
        ckpt_path = os.path.join(args.log_dir, args.net, args.version, "checkpoints")
        ckpt = os.listdir(ckpt_path)[0]
        model = learner.load_from_checkpoint(
            os.path.join(ckpt_path, ckpt), net=net, loss=loss,
            map_location=torch.device('cpu')
        )
        model = model.eval()

    return model
