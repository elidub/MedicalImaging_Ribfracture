from src.model.models import DummyNetwork, UNet3D
from src.model.learner import Learner

def setup_model(net = 'dummy'):
    if net == 'dummy':
        net = DummyNetwork()
    elif net == 'unet3d':
        net = UNet3D(1, 1)
    else:
        raise ValueError('Invalid network type')
    
    model = Learner(net)
    
    return model