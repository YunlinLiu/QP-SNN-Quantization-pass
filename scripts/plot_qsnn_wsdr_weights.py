#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import argparse
import math

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter

# ensure project root on path and register Spikingformer
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'Spikingformer' / 'cifar10-dvs'))
import model_wsdr as model_def  # registers 'Spikingformer'
from timm.models import create_model

# quant pass & QSNN layers for type checks
from chop.passes.module.transforms import quantize_module_transform_pass
# 尽量避免路径差异导致的 isinstance 失败：优先按类名匹配
try:
    from chop.nn.quantized.modules.spikingformer.conv1d import Conv1dQSNN as _Conv1dQSNN
except Exception:
    _Conv1dQSNN = None
try:
    from chop.nn.quantized.modules.vgg.conv2d import Conv2dQSNN as _Conv2dQSNN
except Exception:
    _Conv2dQSNN = None
from models.layers import MultiStepLIFNodeQCuPy
import copy


def build_quantized_spikingformer():
    m = create_model(
        'Spikingformer',
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )
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
    m, _ = quantize_module_transform_pass(m, copy.deepcopy(pass_args_dvs))
    return m


def collect_qsnn_convs(m: torch.nn.Module):
    layers = []
    for name, module in m.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in {"Conv1dQSNN", "Conv2dQSNN"}:
            layers.append((name, module))
        else:
            # 兜底：如果能够导入真实类，也再判一次 isinstance
            try:
                if (_Conv1dQSNN is not None and isinstance(module, _Conv1dQSNN)) or \
                   (_Conv2dQSNN is not None and isinstance(module, _Conv2dQSNN)):
                    layers.append((name, module))
            except Exception:
                pass
    # 仅选择 block.1 的 q/k/proj/v 四层，保持顺序
    want = ['block.1.attn.q_conv', 'block.1.attn.k_conv', 'block.1.attn.proj_conv', 'block.1.attn.v_conv']
    picked = []
    for w in want:
        for n, mod in layers:
            if n == w:
                picked.append((n, mod))
                break
    if len(picked) == 0:
        return layers
    return picked


def compute_pw_from_weights(w: torch.Tensor, eps: float = 1e-8):
    mu = w.mean()
    sigma = torch.clamp(w.std(unbiased=False), min=eps)
    w_hat = (w - mu) / sigma
    bw = torch.sign(w_hat)
    num_pos = (bw > 0).sum().item()
    num_total = bw.numel()
    pw = num_pos / float(num_total)
    num_neg = num_total - num_pos
    return pw, num_neg, num_pos


def _short_layer_label(name: str) -> str:
    # Keep real name but slightly compact dots for readability
    return name


def make_plot(pw_stats, save_path: Path):
    # pw_stats: list of (layer_name, pw, num_neg, num_pos)
    n = len(pw_stats)
    if n <= 4:
        rows, cols = 2, 2
        fig, axes = plt.subplots(rows, cols, figsize=(8.5, 6.0))
        plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.08, wspace=0.35, hspace=0.35)
    else:
        rows, cols = 4, 4
        fig, axes = plt.subplots(rows, cols, figsize=(12, 9))
        # tighter layout: shrink inter-subplot gaps and left margin
        plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.08, wspace=0.35, hspace=0.7)

    # styling similar to paper
    for idx, (layer_name, pw, num_neg, num_pos) in enumerate(pw_stats):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        # bars with leave spaces
        width = 0.16
        # place bars with moderate spacing (0.30 and 0.70) and keep tick labels as 0/1
        x0, x1 = 0.30, 0.70
        ax.bar([x0], [num_neg], color='#f7c6cf', edgecolor='black', hatch='//', width=width)
        ax.bar([x1], [num_pos], color='#c2e0c6', edgecolor='black', hatch='..', width=width)
        ax.set_xticks([x0, x1])
        ax.set_xticklabels(['0', '1'])
        # xlabel: real layer name under subplot
        ax.set_xlabel(_short_layer_label(layer_name), fontsize=7)
        # y scientific scale 1eX with ticks shown to 2 decimals
        ymax = max(num_neg, num_pos)
        order = int(math.floor(math.log10(max(1.0, ymax))))
        scale = 10 ** order
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f'{y/scale:.2f}'))
        # show 1eX label similar to Matplotlib offset text
        ax.text(0.0, 1.02, f'1e{order}', transform=ax.transAxes, fontsize=8)
        # p_w annotation (use subscript)
        # annotation box (3 lines): p_s, \hat{W}^l=0, \hat{W}^l=1
        box_text = (r"$p_{s}:%0.2f$" % pw) + "\n" + r"$\hat{W}^{l}=0$" + "\n" + r"$\hat{W}^{l}=1$"
        ax.text(0.58, 0.92, box_text, transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, boxstyle='round,pad=0.2'))
        # y limit with headroom & margins for whitespace
        ax.set_ylim(0, ymax * 1.5)
        ax.set_xlim(0.0, 1.0)
        ax.margins(x=0.05)
        # ticks formatting
        ax.tick_params(axis='both', labelsize=8)

    # hide any unused axes if n<16 (robustness)
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')

    # no left vertical title
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot WS-DR weight distributions (Fig.4a style)')
    parser.add_argument('--checkpoint', required=True, help='path to checkpoint_*.pth')
    parser.add_argument('--output', default=None, help='output image path (.png)')
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    m = build_quantized_spikingformer()
    state = torch.load(ckpt_path, map_location='cpu')
    sd = state.get('model', state)
    m.load_state_dict(sd, strict=False)

    layers = collect_qsnn_convs(m)
    if len(layers) == 0:
        raise RuntimeError('No QSNN conv layers found. Ensure you built the quantized model before loading the checkpoint.')

    stats = []
    for name, mod in layers:
        w = mod.weight.detach().cpu()
        pw, num_neg, num_pos = compute_pw_from_weights(w)
        stats.append((name, pw, num_neg, num_pos))

    # limit to first 16 layers for a 4x4 grid (matches our model depth)
    stats = stats[:16]

    out_path = Path(args.output) if args.output else (ckpt_path.parent / 'fig_wsdr_weights.png')
    make_plot(stats, out_path)


if __name__ == '__main__':
    main()


