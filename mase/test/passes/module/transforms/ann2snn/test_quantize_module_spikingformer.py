#!/usr/bin/env python3
import sys
from pathlib import Path

import torch
import torch.nn as nn
from chop.passes.module.transforms.snn.ann2snn import ann2snn_module_transform_pass
from chop.passes.module.transforms import quantize_module_transform_pass

project_root = Path(__file__).resolve().parents[6]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "models"))
from Spikingformer import Spikingformer
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

quan_pass_args = {
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
mg, _ = quantize_module_transform_pass(model, quan_pass_args)
print(mg)
