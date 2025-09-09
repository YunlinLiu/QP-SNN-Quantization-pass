#!/usr/bin/env python3
"""
Replicate the WS-DR ablation bar chart style, but with legend
"Bit Width: 1w-8u" and accuracies [78.3, 79.1, 79.9] for
Q-SNN, Q-SNN+WS-DR, FP SNN respectively.

The visual style aims to match the paper figure:
- First two bars (Q-SNN and Q-SNN+WS-DR) use cross-hatch fill
  to indicate low-bit configuration (here 1w-8u)
- Third bar (FP SNN) uses dotted hatch for 32w-32u
- Black edges, white face with subtle pastel hints under hatches
- Title: "Ablation study for the WS-DR"
- Y label: "Accuracy(%)"
"""

from __future__ import annotations

import argparse
from typing import List

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot(save_path: str) -> None:
    # Data
    labels: List[str] = [
        "QP-SNN\nVanilla\nquant",
        "QP-SNN\nReScaW\nquant",
        "Q-SNN",
        "Q-SNN\n+WS-DR",
        "FP SNN",
    ]
    accuracies: List[float] = [77.51, 78.14, 77.97, 78.77, 79.09]

    # Figure and axis
    fig, ax = plt.subplots(figsize=(4.6, 5.0), dpi=300)
    fig.subplots_adjust(top=0.92, bottom=0.26)

    # Bar positions
    x = list(range(len(labels)))
    width = 0.28

    # Colors designed to resemble the original figure's soft palette
    hatch_color = "#8aa0c8"  # bluish tone for low-bit bars (hatch color via edge)
    fp_face = "#dfe8cf"      # light green for FP bar
    qpsnn_face = "#f3d6b6"    # light orange for QP-SNN bars (8w-32u)

    bars = []
    # First two bars: QP-SNN (8w-32u, diagonal hatch)
    for i in range(2):
        b = ax.bar(
            x[i], accuracies[i], width,
            facecolor=qpsnn_face,
            edgecolor="black",
            linewidth=1.0,
            hatch="//",
        )
        bars.append(b[0])

    # Next two bars: Q-SNN (1w-8u, cross-hatch)
    for i in range(2, 4):
        b = ax.bar(
            x[i], accuracies[i], width,
            facecolor="#d7e3f4",
            edgecolor="black",
            linewidth=1.0,
            hatch="xx",
        )
        bars.append(b[0])

    # Final bar: 32w-32u (dotted hatch)
    b3 = ax.bar(
        x[4], accuracies[4], width,
        facecolor=fp_face,
        edgecolor="black",
        linewidth=1.0,
        hatch="..",
    )
    bars.append(b3[0])

    # Annotations above bars
    for xi, acc in zip(x, accuracies):
        ax.text(
            xi, acc + 0.18, f"{acc:.2f}%",
            ha="center", va="bottom", fontsize=10
        )

    # Axes formatting to mimic the style
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy(%)", fontsize=12)
    # 增加左右留白
    ax.set_xlim(-0.8, len(labels) - 0.4)

    # Y range tailored to the provided accuracies
    ymin = 76.0
    ymax = 80.2
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([76, 77, 78, 79, 80])
    ax.tick_params(axis='y', labelsize=10)

    # No title per request
    # ax.set_title("Ablation study for the WS-DR", fontsize=13, pad=10)

    # Legend patches
    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=qpsnn_face, edgecolor="black", hatch="//",
                  label="Bit Width: 8w-32u"),
        Rectangle((0, 0), 1, 1, facecolor="#d7e3f4", edgecolor="black", hatch="xx",
                  label="Bit Width: 1w-8u"),
        Rectangle((0, 0), 1, 1, facecolor=fp_face, edgecolor="black", hatch="..",
                  label="Bit Width: 32w-32u"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot WS-DR ablation bar (1w-8u)")
    parser.add_argument(
        "--output",
        default="/workspace/QP-SNN-Quantization-pass/figs/wsdr_ablation_1w8u.png",
        help="Output image path (.png)",
    )
    args = parser.parse_args()
    plot(args.output)


if __name__ == "__main__":
    main()


