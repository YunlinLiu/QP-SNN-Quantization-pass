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
import importlib.util
import copy


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

# Vanilla_quan_pass_args = {
#     "by": "regex_name",
#     r"block\.\d+\.attn\.(q_conv|k_conv|v_conv|proj_conv)": {
#         "config": {
#             "name": "vanilla",
#             "num_bits": 8,
#         }
#     },
#     r"block\.\d+\.mlp\.mlp[12]_conv": {
#         "config": {
#             "name": "vanilla",
#             "num_bits": 8,
#         }
#     },
# }

mg_rescaw, _ = quantize_module_transform_pass(model, copy.deepcopy(ReScaW_quan_pass_args))
print(mg_rescaw)

# ---- Quantize cifar10-dvs variant ----
dvs_model_path = project_root / "Spikingformer" / "cifar10-dvs" / "model.py"
spec = importlib.util.spec_from_file_location("spikingformer_cifar10_dvs_model", str(dvs_model_path))
assert spec is not None
dvs_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dvs_module
assert spec.loader is not None
spec.loader.exec_module(dvs_module)

model_dvs = create_model(
    'Spikingformer',
    pretrained=False,
    drop_rate=0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size_h=128, img_size_w=128,
    T=4,
)

for param in model_dvs.parameters():
    param.requires_grad = True

mg_rescaw_dvs, _ = quantize_module_transform_pass(model_dvs, copy.deepcopy(ReScaW_quan_pass_args))
print(mg_rescaw_dvs)

# ---- Quantize dvs128-gesture variant ----
dvs128_model_path = project_root / "Spikingformer" / "dvs128-gesture" / "model.py"
spec = importlib.util.spec_from_file_location("spikingformer_dvs128_gesture_model", str(dvs128_model_path))
assert spec is not None
dvs128_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dvs128_module
assert spec.loader is not None
spec.loader.exec_module(dvs128_module)

model_dvs128 = create_model(
    'Spikingformer',
    pretrained=False,
    drop_rate=0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size_h=128, img_size_w=128,
    T=16,
)

for param in model_dvs128.parameters():
    param.requires_grad = True

mg_rescaw_dvs128, _ = quantize_module_transform_pass(model_dvs128, copy.deepcopy(ReScaW_quan_pass_args))
print(mg_rescaw_dvs128)

# Build a fresh model to avoid in-place replacement accumulation
# model_v = create_model(
#     'Spikingformer',
#     pretrained=False,
#     drop_rate=0,
#     drop_path_rate=0.1,
#     drop_block_rate=None,
#     img_size_h=32, img_size_w=32,
#     patch_size=4, embed_dims=384, num_heads=12, mlp_ratios=4,
#     in_channels=3, num_classes=10, qkv_bias=False,
#     norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
#     T=4,
# )
# for param in model_v.parameters():
#     param.requires_grad = True

# mg_vanilla, _ = quantize_module_transform_pass(model_v, Vanilla_quan_pass_args)
# print(mg_vanilla)

# ---- Quantize TinyImageNet variant ----
tiny_model_path = project_root / "Spikingformer" / "imagenet" / "model.py"
spec_tiny = importlib.util.spec_from_file_location("spikingformer_tiny_model", str(tiny_model_path))
assert spec_tiny is not None
tiny_module = importlib.util.module_from_spec(spec_tiny)
sys.modules[spec_tiny.name] = tiny_module
assert spec_tiny.loader is not None
spec_tiny.loader.exec_module(tiny_module)

model_tiny = create_model(
    'Spikingformer',
    pretrained=False,
    drop_rate=0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size_h=64, img_size_w=64,
    patch_size=4, embed_dims=256, num_heads=4, mlp_ratios=4,
    in_channels=3, num_classes=200, qkv_bias=False,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=8, sr_ratios=1,
    T=4,
)

for param in model_tiny.parameters():
    param.requires_grad = True

mg_rescaw_tiny, _ = quantize_module_transform_pass(model_tiny, copy.deepcopy(ReScaW_quan_pass_args))
print(mg_rescaw_tiny)
