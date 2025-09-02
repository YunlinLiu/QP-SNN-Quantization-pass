#!/usr/bin/env python3
# This example converts a VGG model to quantized version using ReScaW
import sys
from pathlib import Path

import torch
import torch.nn as nn
from chop.passes.module.transforms.snn.ann2snn import ann2snn_module_transform_pass
from chop.passes.module.transforms import quantize_module_transform_pass

# Add project root to import pure_vgg model
project_root = Path(__file__).resolve().parents[7]
sys.path.append(str(project_root))  # For models.layers import in pure_vgg
sys.path.append(str(project_root / "models"))  # For pure_vgg / pure_vggsnn import
from pure_vgg import vgg_16_bn
from pure_vggsnn import vggsnn

vgg = vgg_16_bn(compress_rate=[0.0] * 16, num_classes=10)
for param in vgg.parameters():
    param.requires_grad = True  # QAT training

# def test_ann2snn_module_transform_pass():
ReScaW_quan_pass_args = {
    "by": "regex_name",
    # Quantize all Conv2d layers except the first one (convbn0)
    r"features\.convbn(?!0\b)\d+\.layer\.module": {
        "config": {
            "name": "rescaw",
            "num_bits": 4,
        }
    },
}
mg_rescaw, _ = quantize_module_transform_pass(vgg, ReScaW_quan_pass_args)
print(mg_rescaw)

# Build a fresh model for Vanilla to avoid in-place replacement accumulation
# vgg_v = vgg_16_bn(compress_rate=[0.0] * 16, num_classes=10)
# for param in vgg_v.parameters():
#     param.requires_grad = True  # QAT training

# Vanilla_quan_pass_args = {
#     "by": "regex_name",
#     r"features\.convbn(?!0\b)\d+\.layer\.module": {
#         "config": {
#             "name": "vanilla",
#             "num_bits": 4,
#         }
#     },
# }
# mg_vanilla, _ = quantize_module_transform_pass(vgg_v, Vanilla_quan_pass_args)
# print(mg_vanilla)

# convert_pass_args = {
#     "by": "regex_name",
#     # Add SNN conversion rules here if needed
# }
# mg, _ = ann2snn_module_transform_pass(mg, convert_pass_args)
# ----- VGGSNN (DVS-CIFAR10) with ReScaW quantization -----
vggs = vggsnn(compress_rate=[0.0] * 8, num_classes=10)
for param in vggs.parameters():
    param.requires_grad = True  # QAT training

ReScaW_quan_pass_args_vggsnn = {
    "by": "regex_name",
    # Quantize all Conv2d layers except the first one (convbn0)
    r"features\.convbn(?!0\b)\d+\.layer\.module": {
        "config": {
            "name": "rescaw",
            "num_bits": 8,
        }
    },
}

mg_rescaw_vggsnn, _ = quantize_module_transform_pass(vggs, ReScaW_quan_pass_args_vggsnn)
print(mg_rescaw_vggsnn)
