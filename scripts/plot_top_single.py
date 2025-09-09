#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATASET_ORDER = [
    "CIFAR-10",
    "CIFAR-100",
    "ImageNet",
    "DVS-Gesture",
    "DVS-CIFAR10",
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def arch_group(arch: str) -> str:
    if arch in {"VGG16", "VGGSNN"}:
        return "VGG(VGGSNN)"
    if "ResNet" in arch:
        return "ResNet-20"
    if "Spikingformer" in arch:
        return "Spikingformer"
    return arch


def _extract_last_float(s: str) -> float | None:
    if not isinstance(s, str):
        return None
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", s)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except Exception:
        return None


def _normalize_method(name: str) -> str:
    name = str(name).strip()
    if name in {"Full-Precision SNN", "FP", "FPS"}:
        return "FPS"
    return name


def load_top_csv(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, comment="#")
    lower = {c.lower(): c for c in raw.columns}
    # 模式A：精简top CSV（dataset, architecture, method, bitwidth, size_mb, accuracy）
    if {"dataset", "architecture", "method", "bitwidth", "accuracy"}.issubset(set(lower.keys())):
        df = raw.rename(columns={lower[k]: k for k in ["dataset", "architecture", "method", "bitwidth", "accuracy"]}).copy()
        if "size_mb" in lower:
            df.rename(columns={lower["size_mb"]: "size_mb"}, inplace=True)
        df = df.dropna(subset=["accuracy"])  # 保留有精度的点
        df["dataset"] = df["dataset"].astype(str)
        df["architecture"] = df["architecture"].astype(str)
        df["method"] = df["method"].map(_normalize_method)
        df["bitwidth"] = df["bitwidth"].astype(str)
        df["size_mb"] = pd.to_numeric(df.get("size_mb"), errors="coerce")
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
        df["arch_group"] = df["architecture"].map(arch_group)
    # 模式B：表格导出的聚合CSV（如 figs/top_single/table.csv）
    elif set([c.strip().lower() for c in raw.columns]).issuperset({"dataset", "method", "architecture", "bit~width", "accuracy"}):
        # 标准化列名，向前填充空白
        rename = {}
        for c in raw.columns:
            cl = c.strip().lower()
            if cl == "dataset":
                rename[c] = "Dataset"
            elif cl == "method":
                rename[c] = "Method"
            elif cl == "architecture":
                rename[c] = "Architecture"
            elif cl == "bit~width":
                rename[c] = "Bit~Width"
            elif cl == "accuracy":
                rename[c] = "Accuracy"
            elif cl == "size (mb)":
                rename[c] = "Size (MB)"
        raw.rename(columns=rename, inplace=True)
        for col in ["Dataset", "Method", "Architecture"]:
            if col in raw.columns:
                raw[col] = raw[col].ffill()
        rows: List[Dict[str, str | float]] = []
        for _, r in raw.iterrows():
            dataset = str(r.get("Dataset", "")).strip()
            method = _normalize_method(str(r.get("Method", "")).strip())
            arch = str(r.get("Architecture", "")).strip()
            accs = str(r.get("Accuracy", "")).strip()
            if method not in {"FPS", "QP-SNN", "Q-SNN"}:
                continue
            acc = _extract_last_float(accs)
            if acc is None:
                continue
            if method == "FPS":
                bw = str(r.get("Bit~Width", "32w-32u")).strip() or "32w-32u"
            elif method == "QP-SNN":
                bw = "8w-32u"
            else:
                bw = "1w-8u"
            rows.append({
                "dataset": dataset,
                "architecture": arch,
                "method": method,
                "bitwidth": bw,
                "size_mb": _extract_last_float(str(r.get("Size (MB)", ""))),
                "accuracy": acc,
            })
        df = pd.DataFrame(rows)
        df["arch_group"] = df["architecture"].map(arch_group)
    else:
        raise ValueError("CSV列名不匹配：请提供q_top_results.csv或表格导出的table.csv格式")

    # 保留 三类方法 & 顶配位宽
    allowed = {("FPS", "32w-32u"), ("QP-SNN", "8w-32u"), ("Q-SNN", "1w-8u")}
    if {"method", "bitwidth"}.issubset(df.columns):
        df = df[df[["method", "bitwidth"]].apply(tuple, axis=1).isin(allowed)].copy()
    # 数据集顺序
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    df = df.dropna(subset=["dataset"])  # 去除无效
    return df


def draw_single_multiline(df: pd.DataFrame, out_path: Path) -> None:
    sns.set_style("whitegrid")
    # 增高画布，保持同一竖线对齐
    plt.figure(figsize=(10.2, 9.2))

    # 颜色/标记按方法，线型按架构
    color_map: Dict[str, str] = {
        "FPS": "#1f77b4",      # blue for FP32
        "QP-SNN": "#ff7f0e",   # orange for QP-SNN-8w32u
        "Q-SNN": "#7a7a7a",    # grey for Q-SNN-1w8u
    }
    marker_map: Dict[str, str] = {
        "FPS": "o",
        "QP-SNN": "^",
        "Q-SNN": "s",
    }
    linestyle_map: Dict[str, str] = {
        "VGG(VGGSNN)": "-.",    # VGG dash-dot
        "ResNet-20": "--",
        "Spikingformer": "-",    # Spikingformer solid
    }
    # 方法显示名（仅用于图例标签）
    display_map: Dict[str, str] = {
        "FPS": "FP32",
        "QP-SNN": "QP-SNN-8w32u",
        "Q-SNN": "Q-SNN-1w8u",
    }

    # 确保x刻度顺序
    x_labels = DATASET_ORDER
    x_index = {d: i for i, d in enumerate(x_labels)}

    # 为每个 (method, arch_group) 画一条线（仅VGG与Spikingformer）
    combos = []
    for method in ["FPS", "QP-SNN", "Q-SNN"]:
        for ag in ["VGG(VGGSNN)", "Spikingformer"]:
            combos.append((method, ag))

    plotted_handles = []
    plotted_labels = []
    for method, ag in combos:
        sub = df[(df["method"] == method) & (df["arch_group"] == ag)]
        if sub.empty:
            continue
        sub_sorted = sub.sort_values("dataset")
        xs: List[int] = [x_index[str(d)] for d in sub_sorted["dataset"].tolist()]
        ys: List[float] = sub_sorted["accuracy"].tolist()
        line, = plt.plot(
            xs,
            ys,
            label=f"{method} - {ag}",
            color=color_map.get(method, None),
            linestyle=linestyle_map.get(ag, "-"),
            marker=marker_map.get(method, "o"),
            markersize=8,
            markeredgewidth=1.8,
            markeredgecolor=color_map.get(method, None),
            linewidth=1.8,
        )
        plotted_handles.append(line)
        plotted_labels.append(f"{display_map.get(method, method)} - {ag}")
        # 不在点上标注数值，避免重叠

    plt.xticks(range(len(x_labels)), x_labels, rotation=0)
    plt.ylabel("Accuracy (%)")
    plt.ylim(50, 100)
    # 竖向淡分隔：静态(前三) 与 事件(后两)
    plt.axvline(2.5, color="#dddddd", linestyle=":", linewidth=1)

    # 内嵌图例：仅六条线，放左下角，避免顶部任何文字
    legend = plt.legend(
        handles=plotted_handles,
        labels=plotted_labels,
        ncol=1,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.04),
        frameon=True,
        fancybox=True,
        framealpha=0.90,
        fontsize=10,
    )
    legend.get_frame().set_edgecolor("#cccccc")
    legend.get_frame().set_facecolor("#ffffff")

    # 无标题
    plt.tight_layout(rect=[0, 0, 1, 1])
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=False,
                        default=str(Path(__file__).resolve().parents[1] / "data" / "q_top_results.csv"))
    parser.add_argument("--out", type=str, required=False,
                        default=str(Path(__file__).resolve().parents[1] / "figs" / "top_single" / "top_single_accuracy.png"))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    df = load_top_csv(csv_path)
    draw_single_multiline(df, out_path)


if __name__ == "__main__":
    main()


