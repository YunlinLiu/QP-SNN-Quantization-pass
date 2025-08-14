#!/usr/bin/env python3
# This example converts a ResNet model to quantized version using ReScaW
import sys
from pathlib import Path

import torch
import torch.nn as nn
from chop.passes.module.transforms import quantize_module_transform_pass

# Add project root to import pure_resnet model
project_root = Path(__file__).resolve().parents[6]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "models"))
from pure_resnet import resnet_20

# Create model
resnet = resnet_20(compress_rate=[0.0]*12, num_classes=10)
for param in resnet.parameters():
    param.requires_grad = True  # QAT training

# Quantization configuration
quan_pass_args = {
    "by": "regex_name",
    # Quantize Conv2d layers inside tdLayer (conv1_s, conv2_s), exclude the first conv1
    r"layer\d+\.\d+\.conv[12]_s\.layer\.module$": {
        "config": {
            "name": "rescaw",
            "num_bits": 8,
        }
    },
}

# Apply quantization pass
mg, _ = quantize_module_transform_pass(resnet, quan_pass_args)
print(mg)
