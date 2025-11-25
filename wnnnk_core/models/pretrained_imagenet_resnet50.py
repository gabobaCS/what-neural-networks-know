import torch
from torchvision.models import resnet50, ResNet50_Weights

def pretrained_imagenet_resnet50():
    """
    Loads a pretrained ResNet-50 on ImageNet-1K.

    Returns:
        model (torch.nn.Module): Pretrained ResNet-50 in eval mode.
        weights (ResNet50_Weights): Weights object with metadata.
        transforms (callable): Preprocessing pipeline for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device).eval()
    transforms = weights.transforms()

    return model, weights, transforms
