#!/usr/bin/env python3
# This example converts a VGG model to quantized version using ReScaW
import sys
from pathlib import Path

import torch
import torch.nn as nn
from chop.passes.module.transforms.snn.ann2snn import ann2snn_module_transform_pass
from chop.passes.module.transforms import quantize_module_transform_pass

# Add project root to import pure_vgg model
project_root = Path(__file__).resolve().parents[6]
sys.path.append(str(project_root))  # For models.layers import in pure_vgg
sys.path.append(str(project_root / "models"))  # For pure_vgg import
from pure_vgg import vgg_16_bn

vgg = vgg_16_bn(compress_rate=[0.0] * 16, num_classes=10)
for param in vgg.parameters():
    param.requires_grad = True  # QAT training

# def test_ann2snn_module_transform_pass():
quan_pass_args = {
    "by": "regex_name",
    # Quantize all Conv2d layers except the first one (convbn0)
    r"features\.convbn(?!0\b)\d+\.layer\.module": {
        "config": {
            "name": "rescaw",
            "num_bits": 4,
        }
    },
}
mg, _ = quantize_module_transform_pass(vgg, quan_pass_args)
print(mg)

# convert_pass_args = {
#     "by": "regex_name",
#     # Add SNN conversion rules here if needed
# }
# mg, _ = ann2snn_module_transform_pass(mg, convert_pass_args)
