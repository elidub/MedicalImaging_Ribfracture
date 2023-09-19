from src.model.models import DummyCNN, DummyNetwork
from src.model.learner import Learner

def setup_model():
    net = DummyNetwork()
    model = Learner(net)
    return model