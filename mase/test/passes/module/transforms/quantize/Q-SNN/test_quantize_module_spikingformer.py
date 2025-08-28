#!/usr/bin/env python3
import sys
from pathlib import Path

import torch
import torch.nn as nn
from chop.passes.module.transforms import quantize_module_transform_pass

project_root = Path(__file__).resolve().parents[7]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "Spikingformer" / "cifar10"))
# 显式导入 Spikingformer/cifar10/model.py 以触发 timm 注册
import model as spikingformer_model
from timm.models import create_model
from functools import partial
from models.layers import MultiStepLIFNodeQ


def main():
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

    for p in model.parameters():
        p.requires_grad = True

    pass_args = {
        "by": "regex_name",
        "manual_instantiate": True,
        "custom_module_map": {"lifspike_q": MultiStepLIFNodeQ},
        # weights
        r"patch_embed\.block0_conv$": {"config": {"name": "apot", "num_bits": 8, "base_k": 2}},
        r"patch_embed\.block[1-4]_conv$": {"config": {"name": "qsnn"}},
        r"block\.\d+\.attn\.(q_conv|k_conv|v_conv|proj_conv)$": {"config": {"name": "qsnn"}},
        r"block\.\d+\.mlp\.mlp[12]_conv$": {"config": {"name": "qsnn"}},
        r"head$": {"config": {"name": "apot", "num_bits": 8, "base_k": 2}},
        # voltage (T=4 not changed)
        r"patch_embed\.block[1-4]_lif$": {"config": {"name": "lifspike_q", "num_bits": 8}},
        r"block\.\d+\.attn\.attn_lif$": {"config": {"name": "lifspike_q", "num_bits": 8, "thresh": 0.5}},
        r"block\.\d+\.attn\.(proj_lif|q_lif|k_lif|v_lif)$": {"config": {"name": "lifspike_q", "num_bits": 8}},
        r"block\.\d+\.mlp\.(mlp1_lif|mlp2_lif)$": {"config": {"name": "lifspike_q", "num_bits": 8}},
    }

    mg, _ = quantize_module_transform_pass(model, pass_args)
    print(mg)


if __name__ == "__main__":
    main()


