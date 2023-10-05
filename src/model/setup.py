from torch import nn

from src.model.models import DummyNetwork, UNet3D
from src.model.learner import Learner

def setup_model(net = 'dummy'):
    if net == 'dummy':
        net = DummyNetwork()
        loss = nn.CrossEntropyLoss()
    elif net == 'unet3d':
        net = UNet3D(1, 1)
        loss = nn.CrossEntropyLoss()
    else:
        raise ValueError('Invalid network type')
    
    model = Learner(net, loss)
    
    return model