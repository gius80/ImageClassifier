from torchvision import models
from collections import OrderedDict
from torch import nn

def create_model(arch, hidden_units):
    """
    Creates NN model
    Parameters:
    - arch
    - hidden_units
    Return
    - NN model
    """
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    return model
    