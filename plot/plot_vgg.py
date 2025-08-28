#!/usr/bin/env python3
"""
Plot conv-layer weight histograms for VGG-16 (TinyImageNet):
- Vanilla_q8 (raw weights)
- ReScaW_q8 (scaled view, fixed axis)
- Overlay comparison (ReScaW vs Vanilla)
"""
from pathlib import Path
import re
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import logging

# Silence
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# IO
RUN_DIR_VAN = PROJECT_ROOT / 'output_vgg_quan' / 'QP-SNN' / 'TinyImageNet' / 'Vanilla_q8'
RUN_DIR_RES = PROJECT_ROOT / 'output_vgg_quan' / 'QP-SNN' / 'TinyImageNet' / 'ReScaW_q8'
OUT_VAN = PROJECT_ROOT / 'plot' / 'vgg' / 'vgg_Vanilla_q8_TinyImageNet_all_conv_hists.png'
OUT_RES = PROJECT_ROOT / 'plot' / 'vgg' / 'vgg_ReScaW_q8_TinyImageNet_all_conv_hists.png'
OUT_OVERLAY = PROJECT_ROOT / 'plot' / 'vgg' / 'vgg_q8.png'


def _find_ckpt(run_dir: Path) -> Path | None:
    cand = [
        run_dir / 'model_best.pth.tar',
        run_dir / 'last.pth.tar',
    ]
    for p in cand:
        if p.exists():
            return p
    tars = sorted(run_dir.glob('checkpoint-*.pth.tar'))
    if tars:
        return tars[-1]
    tars = sorted(run_dir.glob('*.pth.tar'))
    if tars:
        return tars[-1]
    return None


def _load_state_dict(path: Path) -> dict:
    obj = torch.load(str(path), map_location='cpu')
    if isinstance(obj, dict):
        if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
            return obj['state_dict']
        if 'model' in obj and isinstance(obj['model'], dict):
            return obj['model']
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        return obj
    raise RuntimeError(f'Cannot parse state_dict from {path}')


# VGG conv weight key pattern examples (aligned with our training code)
_VGG_RE = re.compile(r'^(features\.(?:convbn\d+|conv\d+)\.layer\.module|features\.conv\d+\.weight)\.weight$')


def _collect_conv_weight_keys(sd: dict[str, torch.Tensor]) -> list[str]:
    # VGG keys collected by heuristic: 4D conv weights under features.*
    keys: list[str] = []
    for k, v in sd.items():
        if k.endswith('.weight') and isinstance(v, torch.Tensor) and v.ndim == 4 and k.startswith('features'):
            keys.append(k)
    keys.sort()
    return keys


def _short_title_from_key(key: str) -> str:
    # e.g., features.convbn10.layer.module.weight -> conv10
    parts = key.split('.')
    # try to locate conv name piece
    name = None
    for p in parts:
        if p.startswith('conv'):
            name = p
            break
        if p.startswith('convbn'):
            name = p
            break
    return name or parts[-4]


def plot_panel_for_run(run_dir: Path, out_path: Path,
                       plot_rescaw_scaled: bool = False,
                       fixed_xlim: tuple | None = None):
    ckpt = _find_ckpt(run_dir)
    if ckpt is None:
        return
    sd = _load_state_dict(ckpt)

    conv_keys = _collect_conv_weight_keys(sd)
    if len(conv_keys) == 0:
        return

    num = len(conv_keys)
    cols = 6
    rows = (num + cols - 1) // cols
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*3.0, rows*2.6))
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    axes = np.array(axes)

    for idx, key in enumerate(conv_keys):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        w_raw = sd[key].detach().cpu().numpy().ravel()

        if plot_rescaw_scaled:
            gamma = float(np.mean(np.abs(w_raw))) if w_raw.size > 0 else 1.0
            gamma = max(gamma, 1e-12)
            w_plot = w_raw / gamma
        else:
            w_plot = w_raw

        p01 = float(np.quantile(w_plot, 0.01))
        p99 = float(np.quantile(w_plot, 0.99))
        a = max(abs(p01), abs(p99))

        if fixed_xlim is not None:
            xmin, xmax = fixed_xlim
        else:
            xmax = max(a * 1.1, 1e-6)
            xmin = -xmax

        bins = np.linspace(xmin, xmax, 81)
        ax.hist(w_plot, bins=bins, color='#7ec8e3', edgecolor='white')
        ax.set_xlim(xmin, xmax)
        ax.set_title(_short_title_from_key(key))
        
        if plot_rescaw_scaled:
            p01_raw = float(np.quantile(w_raw, 0.01))
            p99_raw = float(np.quantile(w_raw, 0.99))
            box = (
                f"scaled P0.99={p99:.2f}\n"
                f"scaled P0.01={p01:.2f}\n"
                f"raw P0.99={p99_raw:.2f}\n"
                f"raw P0.01={p01_raw:.2f}"
            )
        else:
            box = f"P0.99(W) = {p99:.2f}\nP0.01(W) = {p01:.2f}"
        ax.text(0.55, 0.90, box, transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.8, boxstyle='round,pad=0.25'))

    total_axes = rows * cols
    for k in range(num, total_axes):
        r, c = divmod(k, cols)
        axes[r, c].axis('off')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)


def main():
    # Figure 1: Vanilla_q8 (raw weights)
    if RUN_DIR_VAN.exists():
        plot_panel_for_run(RUN_DIR_VAN, OUT_VAN, plot_rescaw_scaled=False, fixed_xlim=None)
    # Figure 2: ReScaW_q8 (scaled, fixed axis)
    if RUN_DIR_RES.exists():
        plot_panel_for_run(RUN_DIR_RES, OUT_RES, plot_rescaw_scaled=True, fixed_xlim=(-5.0, 5.0))
    # Figure 3: Overlay (ReScaW vs Vanilla)
    if RUN_DIR_VAN.exists() and RUN_DIR_RES.exists():
        ckpt_v = _find_ckpt(RUN_DIR_VAN)
        ckpt_r = _find_ckpt(RUN_DIR_RES)
        if ckpt_v is not None and ckpt_r is not None:
            sd_v = _load_state_dict(ckpt_v)
            sd_r = _load_state_dict(ckpt_r)
            conv_keys = _collect_conv_weight_keys(sd_v)
            if len(conv_keys) > 0:
                num = len(conv_keys)
                cols = 6
                rows = (num + cols - 1) // cols
                fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*3.0, rows*2.6))
                if rows == 1:
                    axes = np.expand_dims(axes, 0)
                axes = np.array(axes)
                for idx, key in enumerate(conv_keys):
                    r, c = divmod(idx, cols)
                    ax = axes[r, c]
                    w_v = sd_v[key].detach().cpu().numpy().ravel()
                    w_r_full = sd_r[key].detach().cpu().numpy().ravel()
                    gamma = float(np.mean(np.abs(w_r_full))) if w_r_full.size > 0 else 1.0
                    gamma = max(gamma, 1e-12)
                    w_r = w_r_full / gamma
                    xmin, xmax = -5.0, 5.0
                    bins = np.linspace(xmin, xmax, 81)
                    ax.hist(w_r, bins=bins, color='#2a6f97', alpha=0.85, edgecolor='white', label='ReScaW')
                    ax.hist(w_v, bins=bins, color='#7ec8e3', alpha=0.85, edgecolor='white', label='Vanilla')
                    ax.set_xlim(xmin, xmax)
                    ax.set_title(_short_title_from_key(key))
                    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
                    ax.legend(loc='upper right', frameon=True,
                              fontsize=8, handlelength=1.6,
                              borderpad=0.25, labelspacing=0.25)
                total_axes = rows * cols
                for k in range(num, total_axes):
                    r, c = divmod(k, cols)
                    axes[r, c].axis('off')
                plt.tight_layout()
                OUT_OVERLAY.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(OUT_OVERLAY, dpi=220)


if __name__ == '__main__':
    main()
