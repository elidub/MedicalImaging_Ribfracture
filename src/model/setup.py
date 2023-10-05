from torch import nn
import os

from src.model.models import DummyNetwork, UNet3D
from src.model.learner import Learner

def setup_model(args):
    net = args.net
    if net == 'dummy':
        net = DummyNetwork()
        loss = nn.CrossEntropyLoss()
    elif net == 'unet3d':
        net = UNet3D(1, 1)
        loss = nn.BCELoss()
    else:
        raise ValueError('Invalid network type')
    
    if args.train:
        model = Learner(net, loss)
    elif args.predict:
        ckpt_path = os.path.join(args.log_dir, args.net, args.version, 'checkpoints')
        ckpt = os.listdir(ckpt_path)[0]
        model = Learner.load_from_checkpoint(os.path.join(ckpt_path, ckpt), net = net, loss = loss)

    return model
