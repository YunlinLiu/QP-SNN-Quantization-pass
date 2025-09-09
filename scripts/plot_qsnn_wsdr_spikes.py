#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import argparse
import math

import torch
import torch.utils.data
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import patches as mpatches

# project imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'Spikingformer' / 'cifar10-dvs'))
import model_wsdr as model_def
import utils
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import cifar10_dvs
from timm.models import create_model

from chop.passes.module.transforms import quantize_module_transform_pass
from models.layers import MultiStepLIFNodeQCuPy
import copy


def build_quantized_model():
    m = create_model(
        'Spikingformer',
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )
    pass_args = {
        "by": "regex_name",
        "manual_instantiate": True,
        "custom_module_map": {"lifspike_q_cupy": MultiStepLIFNodeQCuPy},
        # weights
        r"patch_embed\.proj_conv$": {"config": {"name": "apot", "num_bits": 8, "base_k": 2}},
        r"patch_embed\.proj[1-4]_conv$": {"config": {"name": "qsnn"}},
        r"block\.\d+\.attn\.(q_conv|k_conv|v_conv|proj_conv)$": {"config": {"name": "qsnn"}},
        r"block\.\d+\.mlp\.mlp[12]_conv$": {"config": {"name": "qsnn"}},
        r"head$": {"config": {"name": "apot", "num_bits": 8, "base_k": 2}},
        # voltage
        r"patch_embed\.proj[1-4]_lif$": {"config": {"name": "lifspike_q_cupy", "num_bits": 8, "tau": 1.5}},
        r"block\.\d+\.attn\.attn_lif$": {"config": {"name": "lifspike_q_cupy", "num_bits": 8, "tau": 1.5, "thresh": 0.5}},
        r"block\.\d+\.attn\.(proj_lif|q_lif|k_lif|v_lif)$": {"config": {"name": "lifspike_q_cupy", "num_bits": 8, "tau": 1.5}},
        r"block\.\d+\.mlp\.(mlp1_lif|mlp2_lif)$": {"config": {"name": "lifspike_q_cupy", "num_bits": 8, "tau": 1.5}},
    }
    m, _ = quantize_module_transform_pass(m, copy.deepcopy(pass_args))
    return m


def collect_target_lif_modules(model: torch.nn.Module):
    names = []
    modules = []
    for n, m in model.named_modules():
        cls = m.__class__.__name__
        if cls in {"MultiStepLIFNode", "MultiStepParametricLIFNode", "MultiStepLIFNodeQ", "MultiStepLIFNodeQCuPy", "LIFSpike", "LIFSpikeQ"}:
            names.append(n)
            modules.append(m)
    # Only pick block.1 q/k/proj/v in fixed order (4 subplots)
    order_keys_4 = [
        'block.1.attn.q_lif',
        'block.1.attn.k_lif',
        'block.1.attn.proj_lif',
        'block.1.attn.v_lif',
    ]
    picked = []
    for key in order_keys_4:
        for n, m in zip(names, modules):
            if n == key:
                picked.append((n, m))
                break
    # fallback by appearance if not found
    if len(picked) < 4:
        seen = set(n for n, _ in picked)
        for n, m in zip(names, modules):
            if n not in seen:
                picked.append((n, m))
            if len(picked) >= 4:
                break
    return picked[:4]


def build_dataloader(dataset_dir: str, T: int, batch_size: int = 16, workers: int = 4):
    origin_set = cifar10_dvs.CIFAR10DVS(root=dataset_dir, data_type='frame', frames_number=T, split_by='number')
    # split the same way as training default (90/10), but for plotting我们用测试子集
    # reuse util from train script
    # 简化：直接使用全部样本以获得更稳定 ps（也可取测试子集）
    dataset = origin_set
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False, pin_memory=True)
    return loader


def plot_fig(ps_stats, count_stats, layer_names, save_path: Path):
    # ps_stats/list float; count_stats/list of (zeros, ones)
    n = len(layer_names)
    if n <= 4:
        rows, cols = 2, 2
        fig, axes = plt.subplots(rows, cols, figsize=(8.5, 6.0))
        plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.08, wspace=0.35, hspace=0.25)
    else:
        rows, cols = 3, 6
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6.8))
        plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.08, wspace=0.35, hspace=0.7)

    for idx in range(n):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        zeros, ones = count_stats[idx]
        width = 0.16
        x0, x1 = 0.30, 0.70
        # colors to mimic paper (left yellow hatched, right blue dotted)
        ax.bar([x0], [zeros], color='#f7e3a3', edgecolor='black', hatch='//', width=width)
        ax.bar([x1], [ones], color='#bcd4f6', edgecolor='black', hatch='..', width=width)
        ax.set_xticks([x0, x1])
        ax.set_xticklabels(['0', '1'])
        ax.set_xlabel(layer_names[idx], fontsize=7)

        ymax = max(zeros, ones)
        order = int(math.floor(math.log10(max(1.0, ymax))))
        scale = 10 ** order
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f'{y/scale:.2f}'))
        ax.text(0.0, 1.02, f'1e{order}', transform=ax.transAxes, fontsize=8)
        ax.set_ylim(0, ymax * 1.5)
        ax.set_xlim(0.0, 1.0)
        ax.margins(x=0.05)
        ax.tick_params(axis='both', labelsize=8)

        ps = ps_stats[idx]
        box_text = (r"$p_{s}:%0.2f$" % ps) + "\n" + r"$S^{l}=0$" + "\n" + r"$S^{l}=1$"
        ax.text(0.58, 0.92, box_text, transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, boxstyle='round,pad=0.2'))

    # hide extra axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')

    # remove left vertical title as requested
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved to {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Plot WS-DR spike distributions (Fig.4b style, selectable layers)')
    parser.add_argument('--checkpoint', required=True, help='path to checkpoint_*.pth')
    parser.add_argument('--data-path', default='/workspace/QP-SNN-Quantization-pass/data/CIFAR10DVS')
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', default=None, help='optional output png path')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # build model + quant pass, load checkpoint
    model = build_quantized_model()
    state = torch.load(args.checkpoint, map_location='cpu')
    sd = state.get('model', state)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    # collect LIF layers (18)
    layer_pairs = collect_target_lif_modules(model)
    layer_names = [n for n, _ in layer_pairs]

    # allocate counters
    ones = [0] * len(layer_pairs)
    total = [0] * len(layer_pairs)

    # register hooks
    handles = []
    def make_hook(i):
        def hook(_m, _inp, out):
            s = out.detach()
            ones[i] += (s > 0).sum().item()
            total[i] += s.numel()
        return hook
    for i, (_, m) in enumerate(layer_pairs):
        handles.append(m.register_forward_hook(make_hook(i)))

    # dataloader
    loader = build_dataloader(args.data_path, args.T, args.batch_size, args.workers)

    with torch.no_grad():
        for images, _targets in loader:
            images = images.float().to(device)
            _ = model(images)
            functional.reset_net(model)

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    ps = [(o / t) if t > 0 else 0.0 for o, t in zip(ones, total)]
    counts = [(t - o, o) for o, t in zip(ones, total)]

    out_png = Path(args.output) if args.output else (Path(args.checkpoint).parent / 'fig_wsdr_spikes.png')
    plot_fig(ps, counts, layer_names, out_png)

    # save csv
    out_csv = Path(args.checkpoint).parent / 'fig_wsdr_spikes_stats.csv'
    with open(out_csv, 'w') as f:
        f.write('layer,zeros,ones,total,ps\n')
        for (name, (z, o), t, p) in zip(layer_names, counts, total, ps):
            f.write(f'{name},{z},{o},{t},{p:.6f}\n')
    print(f'Saved stats to {out_csv}')


if __name__ == '__main__':
    main()


