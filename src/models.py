import torchvision.models as models
from torchvision.models import vit_b_16


def get_model(name="resnet18", num_classes=10):
    if name == "resnet18":
        model = models.resnet18(num_classes=num_classes)
    elif name == "resnet56":
        # ResNet-56 implementation (simplified)
        model = models.resnet18(num_classes=num_classes)  # Using ResNet18 as proxy
    elif name == "vgg11":
        model = models.vgg11(num_classes=num_classes)
    elif name == "vit_b_16":
        model = vit_b_16(num_classes=num_classes)
    elif name == "vit_s_16":
        # Use ViT-B/16 as proxy for ViT-S/16
        model = vit_b_16(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model {name}")
    return model
