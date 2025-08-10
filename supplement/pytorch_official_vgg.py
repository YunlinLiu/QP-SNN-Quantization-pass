"""
PyTorch Official VGG Implementation
Source: https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html

This is the official torchvision VGG implementation, which serves as the standard reference
for creating SNN (Spiking Neural Network) versions of VGG architectures.
"""

from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn

# Note: In actual torchvision, these imports would be from torchvision modules
# from ..transforms._presets import ImageClassification
# from ..utils import _log_api_usage_once
# from ._api import register_model, Weights, WeightsEnum
# from ._meta import _IMAGENET_CATEGORIES
# from ._utils import _ovewrite_named_param, handle_legacy_interface

__all__ = [
    "VGG",
    "VGG11_Weights",
    "VGG11_BN_Weights", 
    "VGG13_Weights",
    "VGG13_BN_Weights",
    "VGG16_Weights",
    "VGG16_BN_Weights",
    "VGG19_Weights", 
    "VGG19_BN_Weights",
    "vgg11",
    "vgg11_bn",
    "vgg13", 
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]


class VGG(nn.Module):
    """
    VGG model architecture from "Very Deep Convolutional Networks For Large-Scale Image Recognition"
    
    Args:
        features (nn.Module): Features layers (convolutional part)
        num_classes (int): Number of classes for classification
        init_weights (bool): Whether to initialize weights
        dropout (float): Dropout rate in classifier
    """
    
    def __init__(
        self, 
        features: nn.Module, 
        num_classes: int = 1000, 
        init_weights: bool = True, 
        dropout: float = 0.5
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)  # Original torchvision logging
        
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096), 
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming normal for conv layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    """
    Create VGG layers from configuration list
    
    Args:
        cfg: Configuration list where numbers represent output channels, 'M' represents MaxPool
        batch_norm: Whether to add batch normalization
        
    Returns:
        nn.Sequential: Sequential container with VGG layers
    """
    layers: List[nn.Module] = []
    in_channels = 3
    
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            
    return nn.Sequential(*layers)


# VGG configurations
cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],  # VGG11
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],  # VGG13
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],  # VGG16
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],  # VGG19
}


def _vgg(cfg: str, batch_norm: bool, weights: Optional[Any] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """Internal function to create VGG models"""
    if weights is not None:
        kwargs["init_weights"] = False
        # In actual torchvision, weights handling is more complex
        # if weights.meta["categories"] is not None:
        #     _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    
    if weights is not None:
        # In actual torchvision: model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        pass
        
    return model


# Model creation functions

def vgg11(*, weights: Optional[Any] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-11 from `Very Deep Convolutional Networks for Large-Scale Image Recognition`
    
    Args:
        weights: The pretrained weights to use
        progress: If True, displays a progress bar of the download to stderr
        **kwargs: parameters passed to the VGG base class
    """
    return _vgg("A", False, weights, progress, **kwargs)


def vgg11_bn(*, weights: Optional[Any] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-11-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition`"""
    return _vgg("A", True, weights, progress, **kwargs)


def vgg13(*, weights: Optional[Any] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-13 from `Very Deep Convolutional Networks for Large-Scale Image Recognition`"""
    return _vgg("B", False, weights, progress, **kwargs)


def vgg13_bn(*, weights: Optional[Any] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-13-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition`"""
    return _vgg("B", True, weights, progress, **kwargs)


def vgg16(*, weights: Optional[Any] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-16 from `Very Deep Convolutional Networks for Large-Scale Image Recognition`
    
    This is the most commonly used VGG variant, with 16 layers:
    - 13 convolutional layers
    - 3 fully connected layers
    
    Architecture: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    """
    return _vgg("D", False, weights, progress, **kwargs)


def vgg16_bn(*, weights: Optional[Any] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-16-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition`"""
    return _vgg("D", True, weights, progress, **kwargs)


def vgg19(*, weights: Optional[Any] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-19 from `Very Deep Convolutional Networks for Large-Scale Image Recognition`"""
    return _vgg("E", False, weights, progress, **kwargs)


def vgg19_bn(*, weights: Optional[Any] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-19-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition`"""
    return _vgg("E", True, weights, progress, **kwargs)


if __name__ == "__main__":
    """Example usage and testing"""
    
    print("=== PyTorch Official VGG Implementation ===")
    print("\nVGG Configurations:")
    for name, cfg in cfgs.items():
        print(f"  {name}: {cfg}")
    
    print("\nCreating VGG16 model...")
    model = vgg16(num_classes=1000)
    print(f"VGG16 model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(1, 3, 224, 224)  # Standard ImageNet input size
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    print("\nModel structure:")
    print("Features (convolutional part):")
    print(f"  {len(list(model.features.children()))} layers")
    print("Classifier (fully connected part):")
    print(f"  {len(list(model.classifier.children()))} layers")
    
    print("\n=== Ready for SNN conversion! ===") 