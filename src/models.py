import torch.nn as nn
import torchvision.models as models

def get_model(name="resnet18", num_classes=10):
    if name == "resnet18":
        model = models.resnet18(num_classes=num_classes)
    elif name == "vgg11":
        model = models.vgg11(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model {name}")
    return model
