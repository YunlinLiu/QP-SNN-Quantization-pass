#!/usr/bin/env python3
"""
Plot conv-layer weight histograms (counts) of Spikingformer-4-384 on CIFAR-10:
- Non-quantized run (float32)
- Q8 quantized run (ReScaW, 8-bit)

Each figure is a 4x6 panel across 4 transformer blocks × [q,k,v,proj,mlp1,mlp2].
Histogram x-range is per-layer adaptive to a = max(|P0.01|, |P0.99|), with 10% margin.
"""
import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure model registry available
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'models'))
sys.path.append(str(PROJECT_ROOT / 'Spikingformer' / 'cifar10'))

from timm.models import create_model, load_checkpoint
from chop.passes.module.transforms import quantize_module_transform_pass

RUN_DIR_NOTQ = PROJECT_ROOT / 'output_spikingformer_quan' / 'not_quantized' / '20250825-025808-vitsnn-32'
OUT_PATH_NOTQ = PROJECT_ROOT / 'plot' / 'spikingformer_not_quantized_CIFAR10_all_conv_hists.png'
RUN_DIR_Q8 = PROJECT_ROOT / 'output_spikingformer_quan' / 'q8' / '20250825-040444-vitsnn-32'
OUT_PATH_Q8 = PROJECT_ROOT / 'plot' / 'spikingformer_q8_CIFAR10_all_conv_hists.png'


def plot_panel_for_run(run_dir: Path, out_path: Path, apply_quan: bool,
                       plot_rescaw_scaled: bool = False,
                       fixed_xlim: tuple | None = None):
    # Load args
    with open(run_dir / 'args.yaml', 'r') as f:
        args = yaml.safe_load(f)

    # Register Spikingformer
    import model as spikingformer_model  # noqa: F401

    # Build model
    model = create_model(
        'Spikingformer',
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=0.2,
        drop_block_rate=None,
        img_size_h=args['img_size'], img_size_w=args['img_size'],
        patch_size=args['patch_size'], embed_dims=args['dim'], num_heads=args['num_heads'], mlp_ratios=args['mlp_ratio'],
        in_channels=3, num_classes=args['num_classes'], qkv_bias=False,
        depths=args['depths'], sr_ratios=1,
    )

    # For q8: apply quantization transform so that checkpoint keys match
    if apply_quan:
        quan_pass_args = {
            "by": "regex_name",
            r"block\.\d+\.attn\.(q_conv|k_conv|v_conv|proj_conv)": {
                "config": {"name": "rescaw", "num_bits": 8}
            },
            r"block\.\d+\.mlp\.mlp[12]_conv": {
                "config": {"name": "rescaw", "num_bits": 8}
            },
        }
        model, _ = quantize_module_transform_pass(model, quan_pass_args)

    # Load weights
    load_checkpoint(model, str(run_dir / 'model_best.pth.tar'))
    sd = model.state_dict()

    # Layer grid
    blocks = int(args['depths']) if isinstance(args['depths'], int) else 4
    cols = [
        ('attn.q_conv',  'q_conv'),
        ('attn.k_conv',  'k_conv'),
        ('attn.v_conv',  'v_conv'),
        ('attn.proj_conv','proj_conv'),
        ('mlp.mlp1_conv','mlp1_conv'),
        ('mlp.mlp2_conv','mlp2_conv'),
    ]

    fig, axes = plt.subplots(nrows=blocks, ncols=len(cols), figsize=(6*3.0, blocks*2.6))
    if blocks == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(blocks):
        for j, (spec, short) in enumerate(cols):
            ax = axes[i, j]
            key = f'block.{i}.{spec}.weight'
            title = f'Block{i}.{short}'
            if key not in sd:
                ax.set_title(title + ' (missing)')
                ax.axis('off')
                continue
            w_raw = sd[key].detach().cpu().numpy().ravel()

            # Decide data to plot
            if plot_rescaw_scaled:
                # gamma = ||W||_1 / |W| = mean(|W|)
                gamma = float(np.mean(np.abs(w_raw))) if w_raw.size > 0 else 1.0
                gamma = max(gamma, 1e-12)
                w_plot = w_raw / gamma
            else:
                w_plot = w_raw

            # Percentiles for plotted data
            p01 = float(np.quantile(w_plot, 0.01))
            p99 = float(np.quantile(w_plot, 0.99))
            a = max(abs(p01), abs(p99))

            # Axis range
            if fixed_xlim is not None:
                xmin, xmax = fixed_xlim
            else:
                xmax = max(a * 1.1, 1e-6)
                xmin = -xmax

            bins = np.linspace(xmin, xmax, 81)
            ax.hist(w_plot, bins=bins, color='#7ec8e3', edgecolor='white')
            ax.set_xlim(xmin, xmax)
            ax.set_title(title)
            # Annotation text: show both raw and (if enabled) scaled percentiles
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

    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    print(f'[saved] {out_path}')


def main():
    # not-quantized (keep original behavior)
    plot_panel_for_run(RUN_DIR_NOTQ, OUT_PATH_NOTQ, apply_quan=False,
                       plot_rescaw_scaled=False, fixed_xlim=None)
    # q8 quantized (visualize ReScaW scaling; fixed axis to [-2,2])
    plot_panel_for_run(RUN_DIR_Q8, OUT_PATH_Q8, apply_quan=True,
                       plot_rescaw_scaled=True, fixed_xlim=(-5.0, 5.0))


if __name__ == '__main__':
    main()
