#!/usr/bin/env python3
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[5]
sys.path.append(str(project_root / "Spikingformer" / "imagenet"))
sys.path.append(str(project_root))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from timm.models import create_model
from chop.passes.module.transforms import quantize_module_transform_pass
from models.layers import MultiStepLIFNodeQCuPy
import copy

import train as base_train


def create_quantized_model(args):
    args.img_size = 64
    args.num_classes = 200
    model = create_model(
        'Spikingformer',
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0.2,
        drop_block_rate=None,
        img_size_h=args.img_size, img_size_w=args.img_size,
        patch_size=4, embed_dims=256, num_heads=4, mlp_ratios=4,
        in_channels=3, num_classes=args.num_classes, qkv_bias=False,
        depths=8, sr_ratios=1,
        T=4,
    )

    pass_args = {
        "by": "regex_name",
        "manual_instantiate": True,
        "custom_module_map": {"lifspike_q_cupy": MultiStepLIFNodeQCuPy},
        r"patch_embed\.proj_conv$": {"config": {"name": "apot", "num_bits": 8, "base_k": 2}},
        r"patch_embed\.proj[1-4]_conv$": {"config": {"name": "qsnn"}},
        r"block\.\d+\.attn\.(q_conv|k_conv|v_conv|proj_conv)$": {"config": {"name": "qsnn"}},
        r"block\.\d+\.mlp\.mlp[12]_conv$": {"config": {"name": "qsnn"}},
        r"head$": {"config": {"name": "apot", "num_bits": 8, "base_k": 2}},
        # voltage 4-bit
        r"patch_embed\.proj[1-4]_lif$": {"config": {"name": "lifspike_q_cupy", "num_bits": 4, "tau": 1.5}},
        r"block\.\d+\.attn\.attn_lif$": {"config": {"name": "lifspike_q_cupy", "num_bits": 4, "tau": 1.5, "thresh": 0.5}},
        r"block\.\d+\.attn\.(proj_lif|q_lif|k_lif|v_lif)$": {"config": {"name": "lifspike_q_cupy", "num_bits": 4, "tau": 1.5}},
        r"block\.\d+\.mlp\.(mlp1_lif|mlp2_lif)$": {"config": {"name": "lifspike_q_cupy", "num_bits": 4, "tau": 1.5}},
    }
    model, _ = quantize_module_transform_pass(model, copy.deepcopy(pass_args))
    return model


if __name__ == '__main__':
    from timm.models import create_model as _orig_create_model

    def _wrapper_create_model(*_, **__):
        return _orig_create_model(*_, **__)

    base_train.create_model = _wrapper_create_model
    _orig_main = base_train.main

    def _patched_main():
        from timm.utils import setup_default_logging
        setup_default_logging()
        args, args_text = base_train._parse_args()
        args.data_dir = "/data/dataset/tiny-imagenet-200"
        args.dataset = "image_folder"
        args.train_split = "train"
        args.val_split = "val"
        args.device = 'cuda:2'
        model = create_quantized_model(args)
        globals()['__injected_model__'] = model
        def _injector(*a, **k):
            return globals().pop('__injected_model__')
        from timm import models as _timm_models
        _timm_models.create_model, backup = _injector, _timm_models.create_model
        try:
            _orig_main()
        finally:
            _timm_models.create_model = backup

    base_train.main = _patched_main
    base_train.main()


