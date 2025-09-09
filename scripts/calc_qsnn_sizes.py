#!/usr/bin/env python3
import sys
import re
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "models"))

from pure_vgg import vgg_16_bn as build_vgg
from pure_resnet import resnet_20 as build_resnet


def num_params_of(m: nn.Module) -> Tuple[int, int]:
    """Return (weight_params, bias_params) for modules that have them; else (0,0)."""
    w = 0
    b = 0
    if hasattr(m, "weight") and isinstance(getattr(m, "weight"), torch.nn.Parameter):
        w = m.weight.numel()
    if hasattr(m, "bias") and isinstance(getattr(m, "bias"), torch.nn.Parameter) and m.bias is not None:
        b = m.bias.numel()
    return w, b


def count_sizes(model: nn.Module, pass_patterns: Dict[str, str]) -> Dict[str, int]:
    """
    pass_patterns: mapping from pattern_name to regex string identifying modules
      keys: 'w8' (APoT 8b weights), 'w1' (QSNN 1b weights)
    Returns dict with keys: pq1, pq8, pfp (counts of parameters, not bits)
    Notes:
      - All biases (conv/linear), all BatchNorm/LayerNorm weights+biases are treated as FP32 (pfp)
    """
    r_w8 = re.compile(pass_patterns.get("w8", r"^$"))
    r_w1 = re.compile(pass_patterns.get("w1", r"^$"))

    pq1 = 0
    pq8 = 0
    pfp = 0

    for name, module in model.named_modules():
        # BatchNorm / LayerNorm learnable params are FP32
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)):
            if hasattr(module, "weight") and module.weight is not None:
                pfp += module.weight.numel()
            if hasattr(module, "bias") and module.bias is not None:
                pfp += module.bias.numel()
            continue

        # Only consider conv/linear weights for quantization
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            w, b = num_params_of(module)
            if w:
                if r_w8.search(name):
                    pq8 += w
                elif r_w1.search(name):
                    pq1 += w
                else:
                    # anything not matched remains FP32
                    pfp += w
            if b:
                pfp += b

    return {"pq1": pq1, "pq8": pq8, "pfp": pfp}


def bits_to_bytes(bits: int) -> int:
    return (bits + 7) // 8


def calc_size_bytes(counts: Dict[str, int]) -> int:
    bits = counts["pq1"] * 1 + counts["pq8"] * 8 + counts["pfp"] * 32
    return bits_to_bytes(bits)


def fmt(b: int) -> str:
    return f"{b} B ≈ {b/1e6:.3f} MB"


def vgg_qsnn_sizes(num_classes: int) -> Dict[str, int]:
    model = build_vgg(compress_rate=[0.0] * 16, num_classes=num_classes)
    # QSNN pass: convbn0 & classifier -> APoT 8b; other convs -> QSNN 1b
    patterns = {
        "w8": r"^(features\.convbn0\.layer\.module|classifier\.linear1\.module)$",
        "w1": r"^features\.convbn(?!0\b)\d+\.layer\.module$",
    }
    counts = count_sizes(model, patterns)
    return counts


def resnet_qsnn_sizes(num_classes: int) -> Dict[str, int]:
    model = build_resnet(compress_rate=[0.0] * 12, num_classes=num_classes)
    # QSNN pass: stem & fc -> APoT 8b; all block convs -> QSNN 1b
    patterns = {
        "w8": r"^(conv1_s\.layer\.module|fc\.module)$",
        "w1": r"^layer\d+\.\d+\.conv[12]_s\.layer\.module$",
    }
    counts = count_sizes(model, patterns)
    return counts


def spikingformer_qsnn_sizes(dataset: str, num_classes: int) -> Dict[str, int]:
    """
    dataset: one of {'cifar10', 'cifar100', 'imagenet'}
    Note: we reuse CIFAR-10 architecture for CIFAR-100 by only changing num_classes.
    Imagenet variant uses the dedicated model under Spikingformer/imagenet.
    """
    if dataset in ("cifar10", "cifar100"):
        # import CIFAR10 variant directly and instantiate without timm registry
        sys.path.append(str(ROOT / "Spikingformer" / "cifar10"))
        import model as spk_cifar  # type: ignore
        model = spk_cifar.vit_snn(
            img_size_h=32, img_size_w=32, patch_size=4, in_channels=3,
            num_classes=num_classes, embed_dims=384, num_heads=12,
            mlp_ratios=4, qkv_bias=False, drop_rate=0.0, attn_drop_rate=0.0,
            drop_path_rate=0.1, norm_layer=nn.LayerNorm, depths=4, sr_ratios=1, T=4,
        )
        patterns = {
            # weights: block0 & head 8b; others 1b
            "w8": r"^(patch_embed\.block0_conv|head)$",
            "w1": r"^(patch_embed\.block[1-4]_conv|block\.\d+\.attn\.(q_conv|k_conv|v_conv|proj_conv)|block\.\d+\.mlp\.mlp[12]_conv)$",
        }
    elif dataset == "imagenet":
        # import ImageNet/Tiny variant directly and instantiate
        sys.path.append(str(ROOT / "Spikingformer" / "imagenet"))
        import model as spk_imnet  # type: ignore
        model = spk_imnet.vit_snn(
            img_size_h=64, img_size_w=64, patch_size=4, in_channels=3,
            num_classes=num_classes, embed_dims=256, num_heads=4,
            mlp_ratios=4, qkv_bias=False, drop_rate=0.0, attn_drop_rate=0.0,
            drop_path_rate=0.1, norm_layer=nn.LayerNorm, depths=8, sr_ratios=1, T=4,
        )
        patterns = {
            # weights: proj_conv & head 8b; others 1b
            "w8": r"^(patch_embed\.proj_conv|head)$",
            "w1": r"^(patch_embed\.proj[1-4]_conv|block\.\d+\.attn\.(q_conv|k_conv|v_conv|proj_conv)|block\.\d+\.mlp\.mlp[12]_conv)$",
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    counts = count_sizes(model, patterns)
    return counts


def main():
    tasks = [
        ("VGG-16-BN (SNN)", vgg_qsnn_sizes, [("CIFAR-10", 10), ("CIFAR-100", 100), ("ImageNet", 1000)]),
        ("ResNet-20 (SEW SNN)", resnet_qsnn_sizes, [("CIFAR-10", 10), ("CIFAR-100", 100), ("ImageNet", 1000)]),
        ("Spikingformer", None, []),
    ]

    print("== QSNN (1w, 2/4/8u) 模型尺寸，膜电位不计入存储 ==\n")

    for title, fn, cfgs in tasks:
        if fn is not None:
            print(f"[{title}]")
            for ds, nc in cfgs:
                counts = fn(nc)
                size_b = calc_size_bytes(counts)
                print(f"  - {ds}: pq1={counts['pq1']}, pq8={counts['pq8']}, pfp={counts['pfp']} -> {fmt(size_b)}")
            print()

    # Spikingformer separately (architectures differ between datasets)
    print("[Spikingformer]")
    for ds, nc in [("CIFAR-10", 10), ("CIFAR-100", 100), ("ImageNet", 1000)]:
        counts = spikingformer_qsnn_sizes("imagenet" if ds == "ImageNet" else "cifar10", nc)
        size_b = calc_size_bytes(counts)
        print(f"  - {ds}: pq1={counts['pq1']}, pq8={counts['pq8']}, pfp={counts['pfp']} -> {fmt(size_b)}")


if __name__ == "__main__":
    main()


