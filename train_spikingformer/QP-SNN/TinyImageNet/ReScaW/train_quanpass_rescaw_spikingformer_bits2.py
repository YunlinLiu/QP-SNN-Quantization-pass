#!/usr/bin/env python3
import sys
from pathlib import Path

# 复用 imagenet 训练脚本，并在创建模型后插入 ReScaW 量化（2bit）
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root / "Spikingformer" / "imagenet"))
sys.path.append(str(project_root / "mase" / "src"))
sys.path.append(str(project_root))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from timm.models import create_model
from chop.passes.module.transforms import quantize_module_transform_pass
import copy

import train as base_train  # Spikingformer/imagenet/train.py


def create_quantized_model(args):
    # 强制 TinyImageNet 形状/类别（保持最小改动，逻辑不变）
    args.img_size = 64
    args.num_classes = 200
    # 按 imagenet/model 的 Tiny 参数
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
        r"block\.\d+\.attn\.(q_conv|k_conv|v_conv|proj_conv)": {"config": {"name": "rescaw", "num_bits": 2}},
        r"block\.\d+\.mlp\.mlp[12]_conv": {"config": {"name": "rescaw", "num_bits": 2}},
    }
    model, _ = quantize_module_transform_pass(model, copy.deepcopy(rescaw_args))
    return model


if __name__ == '__main__':
    # 直接复用 base_train 入口，但替换模型构建为量化版本
    # 做法：猴子补丁 base_train.create_model 调用链，通过包装 main 前后逻辑最小侵入
    import types
    from timm.models import create_model as _orig_create_model

    def _wrapper_create_model(*_, **__):
        # 该 create_model 不被使用；我们在 main 内部调用前直接构造模型
        return _orig_create_model(*_, **__)

    base_train.create_model = _wrapper_create_model

    # 包装 base_train.main 以在内部使用我们构造的量化模型
    _orig_main = base_train.main

    def _patched_main():
        # 复用其参数解析与训练流程，只在模型创建后替换为量化模型
        from timm.utils import setup_default_logging
        setup_default_logging()
        # 确保使用 TinyImageNet 的 yml（无需手动传 -c）
        if hasattr(base_train, 'config_parser'):
            base_train.config_parser.set_defaults(config=str(project_root / 'Spikingformer' / 'imagenet' / 'imagenet.yml'))
        # 若未显式传入 -c/-config，则注入 yml 路径，避免相对路径查找失败
        import sys as _sys
        if '-c' not in _sys.argv and '--config' not in _sys.argv:
            _sys.argv += ['-c', str(project_root / 'Spikingformer' / 'imagenet' / 'imagenet.yml')]
        args, args_text = base_train._parse_args()
        # TinyImageNet 目录与设备
        args.data_dir = "/data/dataset/tiny-imagenet-200"
        args.dataset = "image_folder"
        args.train_split = "train"
        args.val_split = "val"
        args.device = 'cuda:2'
        # 构建量化模型
        model = create_quantized_model(args)
        # 将构建好的模型注入 base_train.main 的局部流程：通过临时替换 create_model 再次进入
        # 这里采用最小修改：把模型放到全局，供 base_train.main 内部拿到后立即使用
        globals()['__injected_model__'] = model

        # 劫持 timm.create_model 返回已注入模型
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


