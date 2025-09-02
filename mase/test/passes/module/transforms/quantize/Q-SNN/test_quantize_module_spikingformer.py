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
from models.layers import MultiStepLIFNodeQ, MultiStepLIFNodeQCuPy


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
        "custom_module_map": {"lifspike_q": MultiStepLIFNodeQ, "lifspike_q_cupy": MultiStepLIFNodeQCuPy},
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

    # ---- DVS variant (cifar10-dvs) ----
    # import and register DVS model
    import importlib.util
    dvs_model_path = project_root / "Spikingformer" / "cifar10-dvs" / "model.py"
    spec = importlib.util.spec_from_file_location("spikingformer_cifar10_dvs_model_qsnn", str(dvs_model_path))
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

    for p in model_dvs.parameters():
        p.requires_grad = True

    pass_args_dvs = {
        "by": "regex_name",
        "manual_instantiate": True,
        "custom_module_map": {"lifspike_q_cupy": MultiStepLIFNodeQCuPy},
        # weights (Tokenizer naming differs)
        r"patch_embed\.proj_conv$": {"config": {"name": "apot", "num_bits": 8, "base_k": 2}},
        r"patch_embed\.proj[1-4]_conv$": {"config": {"name": "qsnn"}},
        r"block\.\d+\.attn\.(q_conv|k_conv|v_conv|proj_conv)$": {"config": {"name": "qsnn"}},
        r"block\.\d+\.mlp\.mlp[12]_conv$": {"config": {"name": "qsnn"}},
        r"head$": {"config": {"name": "apot", "num_bits": 8, "base_k": 2}},
        # voltage (cupy LIF; enforce tau=1.5; attn_lif thresh=0.5)
        r"patch_embed\.proj[1-4]_lif$": {"config": {"name": "lifspike_q_cupy", "num_bits": 8, "tau": 1.5}},
        r"block\.\d+\.attn\.attn_lif$": {"config": {"name": "lifspike_q_cupy", "num_bits": 8, "tau": 1.5, "thresh": 0.5}},
        r"block\.\d+\.attn\.(proj_lif|q_lif|k_lif|v_lif)$": {"config": {"name": "lifspike_q_cupy", "num_bits": 8, "tau": 1.5}},
        r"block\.\d+\.mlp\.(mlp1_lif|mlp2_lif)$": {"config": {"name": "lifspike_q_cupy", "num_bits": 8, "tau": 1.5}},
    }

    mg_dvs, _ = quantize_module_transform_pass(model_dvs, pass_args_dvs)
    print(mg_dvs)

    # ---- DVS128-Gesture variant (dvs128-gesture) ----
    import importlib.util as _importlib_util
    dvs128_model_path = project_root / "Spikingformer" / "dvs128-gesture" / "model.py"
    spec128 = _importlib_util.spec_from_file_location("spikingformer_dvs128_gesture_model_qsnn", str(dvs128_model_path))
    assert spec128 is not None
    dvs128_module = _importlib_util.module_from_spec(spec128)
    sys.modules[spec128.name] = dvs128_module
    assert spec128.loader is not None
    spec128.loader.exec_module(dvs128_module)

    model_dvs128 = create_model(
        'Spikingformer',
        pretrained=False,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size_h=128, img_size_w=128,
        T=16,
    )

    for p in model_dvs128.parameters():
        p.requires_grad = True

    pass_args_dvs128 = {
        "by": "regex_name",
        "manual_instantiate": True,
        "custom_module_map": {"lifspike_q_cupy": MultiStepLIFNodeQCuPy},
        # weights (Tokenizer naming matches cifar10-dvs)
        r"patch_embed\.proj_conv$": {"config": {"name": "qsnn"}},
        r"patch_embed\.proj[1-4]_conv$": {"config": {"name": "qsnn"}},
        r"block\.\d+\.attn\.(q_conv|k_conv|v_conv|proj_conv)$": {"config": {"name": "qsnn"}},
        r"block\.\d+\.mlp\.mlp[12]_conv$": {"config": {"name": "qsnn"}},
        r"head$": {"config": {"name": "apot", "num_bits": 8, "base_k": 2}},
        # voltage (cupy LIF; enforce tau=1.5; attn_lif thresh=0.5)
        r"patch_embed\.proj[1-4]_lif$": {"config": {"name": "lifspike_q_cupy", "num_bits": 8, "tau": 1.5}},
        r"block\.\d+\.attn\.attn_lif$": {"config": {"name": "lifspike_q_cupy", "num_bits": 8, "tau": 1.5, "thresh": 0.5}},
        r"block\.\d+\.attn\.(proj_lif|q_lif|k_lif|v_lif)$": {"config": {"name": "lifspike_q_cupy", "num_bits": 8, "tau": 1.5}},
        r"block\.\d+\.mlp\.(mlp1_lif|mlp2_lif)$": {"config": {"name": "lifspike_q_cupy", "num_bits": 8, "tau": 1.5}},
    }

    mg_dvs128, _ = quantize_module_transform_pass(model_dvs128, pass_args_dvs128)
    print(mg_dvs128)


if __name__ == "__main__":
    main()

# ---- TinyImageNet variant (QSNN) ----
def main_tiny():
    # import and register TinyImageNet model
    import importlib.util
    project_root = Path(__file__).resolve().parents[7]
    sys.path.append(str(project_root / "Spikingformer" / "imagenet"))
    tiny_model_path = project_root / "Spikingformer" / "imagenet" / "model.py"
    spec = importlib.util.spec_from_file_location("spikingformer_tiny_model_qsnn", str(tiny_model_path))
    assert spec is not None
    tiny_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tiny_module
    assert spec.loader is not None
    spec.loader.exec_module(tiny_module)

    model = create_model(
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

    for p in model.parameters():
        p.requires_grad = True

    pass_args = {
        "by": "regex_name",
        "manual_instantiate": True,
        "custom_module_map": {"lifspike_q_cupy": MultiStepLIFNodeQCuPy},
        # weights (Tokenizer naming for imagenet variant)
        r"patch_embed\.proj_conv$": {"config": {"name": "apot", "num_bits": 8, "base_k": 2}},
        r"patch_embed\.proj[1-4]_conv$": {"config": {"name": "qsnn"}},
        r"block\.\d+\.attn\.(q_conv|k_conv|v_conv|proj_conv)$": {"config": {"name": "qsnn"}},
        r"block\.\d+\.mlp\.mlp[12]_conv$": {"config": {"name": "qsnn"}},
        r"head$": {"config": {"name": "apot", "num_bits": 8, "base_k": 2}},
        # voltage (cupy LIF; keep tau=1.5; attn_lif thresh=0.5)
        r"patch_embed\.proj[1-4]_lif$": {"config": {"name": "lifspike_q_cupy", "num_bits": 8, "tau": 1.5}},
        r"block\.\d+\.attn\.attn_lif$": {"config": {"name": "lifspike_q_cupy", "num_bits": 8, "tau": 1.5, "thresh": 0.5}},
        r"block\.\d+\.attn\.(proj_lif|q_lif|k_lif|v_lif)$": {"config": {"name": "lifspike_q_cupy", "num_bits": 8, "tau": 1.5}},
        r"block\.\d+\.mlp\.(mlp1_lif|mlp2_lif)$": {"config": {"name": "lifspike_q_cupy", "num_bits": 8, "tau": 1.5}},
    }

    mg, _ = quantize_module_transform_pass(model, pass_args)
    print(mg)


