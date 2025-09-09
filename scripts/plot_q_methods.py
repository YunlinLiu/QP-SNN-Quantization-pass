#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment="#")
    # 规范字段
    df["dataset"] = df["dataset"].astype(str)
    df["architecture"] = df["architecture"].astype(str)
    df["method"] = df["method"].astype(str)
    df["bitwidth"] = df["bitwidth"].astype(str)
    df["timestep"] = df["timestep"].astype(int)
    df["size_mb"] = pd.to_numeric(df["size_mb"], errors="coerce")
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    # 去掉缺失精度的行
    df = df.dropna(subset=["accuracy"])
    return df


def order_bitwidth_for_method(method: str) -> List[str]:
    if method == "QP-SNN":
        return ["2w-32u", "4w-32u", "8w-32u"]
    if method == "Q-SNN":
        return ["1w-2u", "1w-4u", "1w-8u"]
    return []


def plot_accuracy_vs_bw(df: pd.DataFrame, out_dir: Path) -> None:
    # 每个 (dataset, architecture) 各画一张：横轴bitwidth(顺序)，纵轴accuracy，曲线为method
    ensure_dir(out_dir)
    grouped = df.groupby(["dataset", "architecture"])
    for (dataset, arch), sub in grouped:
        plt.figure(figsize=(6.2, 4.2))
        sns.set_style("whitegrid")
        # 仅保留两类方法
        sub = sub[sub["method"].isin(["QP-SNN", "Q-SNN"])].copy()
        # 为每个方法设定顺序并画折线
        lines = []
        for method, g in sub.groupby("method"):
            order = order_bitwidth_for_method(method)
            g = g.set_index("bitwidth").reindex(order).reset_index()
            l, = plt.plot(g["bitwidth"], g["accuracy"], marker="o", label=method)
            lines.append(l)
        plt.xlabel("Bit Width configuration")
        plt.ylabel("Accuracy (%)")
        plt.title(f"{dataset} - {arch}")
        if lines:
            plt.legend()
        plt.tight_layout()
        out_path = out_dir / f"acc_vs_bw__{dataset.replace('/', '-') }__{arch.replace('/', '-')}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_acc_vs_size(df: pd.DataFrame, out_dir: Path) -> None:
    # 每个 (dataset, architecture) 各画一张：横轴size(MB)，纵轴accuracy，方法为颜色，位宽为标注
    ensure_dir(out_dir)
    grouped = df.groupby(["dataset", "architecture"])
    for (dataset, arch), sub in grouped:
        # 只看QP-SNN和Q-SNN
        sub = sub[sub["method"].isin(["QP-SNN", "Q-SNN"])].copy()
        if sub.empty:
            continue
        plt.figure(figsize=(6.2, 4.2))
        sns.set_style("whitegrid")
        sns.scatterplot(
            data=sub,
            x="size_mb",
            y="accuracy",
            hue="method",
            style="method",
            s=60,
        )
        # 连线体现同方法不同位宽的帕累托曲线
        for method, g in sub.groupby("method"):
            g_sorted = g.sort_values(["size_mb", "accuracy"])  # 从小到大
            plt.plot(g_sorted["size_mb"], g_sorted["accuracy"], linewidth=1.5, alpha=0.7)
            for _, r in g_sorted.iterrows():
                plt.text(r["size_mb"], r["accuracy"], r["bitwidth"], fontsize=7,
                         ha="left", va="bottom", alpha=0.8)

        plt.xlabel("Model size (MB)")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Size vs Accuracy — {dataset} - {arch}")
        plt.tight_layout()
        out_path = out_dir / f"size_vs_acc__{dataset.replace('/', '-') }__{arch.replace('/', '-')}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "q_methods_results.csv"))
    parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[1] / "figs"))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    df = load_data(csv_path)
    # 生成两类图
    plot_accuracy_vs_bw(df, out_dir / "acc_vs_bw")
    plot_acc_vs_size(df, out_dir / "size_vs_acc")


if __name__ == "__main__":
    main()


