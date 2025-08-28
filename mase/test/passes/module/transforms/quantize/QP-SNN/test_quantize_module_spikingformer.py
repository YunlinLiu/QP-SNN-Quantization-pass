#!/usr/bin/env python3
import sys
from pathlib import Path

import torch
import torch.nn as nn
from chop.passes.module.transforms.snn.ann2snn import ann2snn_module_transform_pass
from chop.passes.module.transforms import quantize_module_transform_pass

project_root = Path(__file__).resolve().parents[7]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "Spikingformer" / "cifar10"))
# 显式导入 Spikingformer/cifar10/model.py 以触发 timm 注册
import model as spikingformer_model
from timm.models import create_model
from functools import partial

model = create_model(
    'Spikingformer',
    pretrained=False,
    drop_rate=0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size_h=32, img_size_w=32,
    patch_size=4, embed_dims=384, num_heads=12, mlp_ratios=4,
    in_channels=3, num_classes=10, qkv_bias=False,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
    T=4,
)

for param in model.parameters():
    param.requires_grad = True

ReScaW_quan_pass_args = {
    "by": "regex_name",
    r"block\.\d+\.attn\.(q_conv|k_conv|v_conv|proj_conv)": {
        "config": {
            "name": "rescaw",
            "num_bits": 8,
        }
    },
    r"block\.\d+\.mlp\.mlp[12]_conv": {
        "config": {
            "name": "rescaw",
            "num_bits": 8,
        }
    },
}

Vanilla_quan_pass_args = {
    "by": "regex_name",
    r"block\.\d+\.attn\.(q_conv|k_conv|v_conv|proj_conv)": {
        "config": {
            "name": "vanilla",
            "num_bits": 8,
        }
    },
    r"block\.\d+\.mlp\.mlp[12]_conv": {
        "config": {
            "name": "vanilla",
            "num_bits": 8,
        }
    },
}

mg_rescaw, _ = quantize_module_transform_pass(model, ReScaW_quan_pass_args)
print(mg_rescaw)

# Build a fresh model to avoid in-place replacement accumulation
model_v = create_model(
    'Spikingformer',
    pretrained=False,
    drop_rate=0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size_h=32, img_size_w=32,
    patch_size=4, embed_dims=384, num_heads=12, mlp_ratios=4,
    in_channels=3, num_classes=10, qkv_bias=False,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
    T=4,
)
for param in model_v.parameters():
    param.requires_grad = True

mg_vanilla, _ = quantize_module_transform_pass(model_v, Vanilla_quan_pass_args)
print(mg_vanilla)
