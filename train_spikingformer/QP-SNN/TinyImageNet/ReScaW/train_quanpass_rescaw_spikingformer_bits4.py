#!/usr/bin/env python3
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root / "Spikingformer" / "imagenet"))
sys.path.append(str(project_root / "mase" / "src"))
sys.path.append(str(project_root))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from timm.models import create_model
from chop.passes.module.transforms import quantize_module_transform_pass
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

    rescaw_args = {
        "by": "regex_name",
        r"block\.\d+\.attn\.(q_conv|k_conv|v_conv|proj_conv)": {"config": {"name": "rescaw", "num_bits": 4}},
        r"block\.\d+\.mlp\.mlp[12]_conv": {"config": {"name": "rescaw", "num_bits": 4}},
    }
    model, _ = quantize_module_transform_pass(model, copy.deepcopy(rescaw_args))
    return model


if __name__ == '__main__':
    import types
    from timm.models import create_model as _orig_create_model

    def _wrapper_create_model(*_, **__):
        return _orig_create_model(*_, **__)

    base_train.create_model = _wrapper_create_model
    _orig_main = base_train.main

    def _patched_main():
        from timm.utils import setup_default_logging
        setup_default_logging()
        if hasattr(base_train, 'config_parser'):
            base_train.config_parser.set_defaults(config=str(project_root / 'Spikingformer' / 'imagenet' / 'imagenet.yml'))
        import sys as _sys
        if '-c' not in _sys.argv and '--config' not in _sys.argv:
            _sys.argv += ['-c', str(project_root / 'Spikingformer' / 'imagenet' / 'imagenet.yml')]
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


