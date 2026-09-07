# -*- coding: utf-8 -*-
"""
步骤2_增强可视化.py
读取三个问题的 csv，生成精美图。不训练模型，几秒出图。
"""

import os
import argparse
import params
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import seaborn as sns
    SNS = True
except Exception:
    SNS = False

OUTPUT_ROOT = params.OUTPUT_EXTENSIONS


def parse_args(argv=None):
    """Parse CLI arguments; falls back to defaults above when none are given."""
    parser = argparse.ArgumentParser(description="QUBO-Fusion enhanced visualization")
    parser.add_argument("--analysis_root", dest="output_root", default=OUTPUT_ROOT,
                        help="Output root produced by three_extension_analyses.py")
    return parser.parse_args(argv)


def apply_config(args):
    global OUTPUT_ROOT, P1_DIR, P2_DIR, P3_DIR, VIZ_DIR
    OUTPUT_ROOT = args.output_root
    P1_DIR = os.path.join(OUTPUT_ROOT, "01_cross_cancer_transfer")
    P2_DIR = os.path.join(OUTPUT_ROOT, "02_within_cancer_cross_dataset")
    P3_DIR = os.path.join(OUTPUT_ROOT, "03_qubo_vs_fullmodel")
    VIZ_DIR = os.path.join(OUTPUT_ROOT, "增强可视化")


_args = parse_args()
apply_config(_args)

os.makedirs(VIZ_DIR, exist_ok=True)

C_QUBO = "#E74C3C"
C_FULL = "#3498DB"

plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 300
def _setup_cjk_font():
    """Pick an available CJK font across platforms so Chinese labels render correctly."""
    try:
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
        for cand in ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC",
                     "WenQuanYi Zen Hei", "Arial Unicode MS"]:
            if cand in available:
                plt.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
                return
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    except Exception:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]


plt.rcParams["axes.unicode_minus"] = False
_setup_cjk_font()
if SNS:
    sns.set_theme(style="white", context="talk")

# 必须放在 set_theme 之后，否则被覆盖
_setup_cjk_font()
plt.rcParams["axes.unicode_minus"] = False


def load(path):
    return pd.read_csv(path) if os.path.exists(path) else None


# ========== 问题一 ==========
def viz_problem1():
    df = load(os.path.join(P1_DIR, "cross_cancer_transfer_results.csv"))
    if df is None:
        print("⚠️ 问题一数据缺失，跳过")
        return

    print("问题一可视化...")

    # 1. 3模型热图网格（QUBO / FullModel 各一张）
    for fset in ["QUBO_Full", "FullModel"]:
        sub_all = df[df["FeatureSet"] == fset]
        if sub_all.empty:
            continue

        mtypes = ["RSF", "BP-Cox", "Fusion"]
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        for ax, mt in zip(axes, mtypes):
            sub = sub_all[sub_all["ModelType"] == mt]
            if sub.empty:
                ax.axis("off")
                continue
            pivot = sub.pivot_table(index="TrainCancer", columns="TestCancer",
                                    values="C_index", aggfunc="mean")
            if SNS:
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                            vmin=0.45, vmax=0.65, ax=ax, cbar_kws={"label": "C-index"},
                            linewidths=0.5, linecolor="white")
            else:
                im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.45, vmax=0.65)
                plt.colorbar(im, ax=ax)
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index)
            ax.set_title(mt, fontsize=15, fontweight="bold")
            ax.set_xlabel("Test Cancer")
            ax.set_ylabel("Train Cancer" if mt == "RSF" else "")

        fig.suptitle(f"跨癌种迁移 C-index ({fset})", fontsize=18, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, f"P1_heatmap_grid_{fset}.png"), bbox_inches="tight")
        plt.close()

    # 2. QUBO vs FullModel 汇总对比
    fusion = df[df["ModelType"] == "Fusion"].copy()
    if not fusion.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        means = fusion.groupby("FeatureSet")["C_index"].agg(["mean", "std"])
        x = np.arange(len(means))
        colors = [C_QUBO if "QUBO" in i else C_FULL for i in means.index]
        ax.bar(x, means["mean"], yerr=means["std"], color=colors, alpha=0.85, capsize=6, width=0.5)
        for i, (m, s) in enumerate(zip(means["mean"], means["std"])):
            ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(means.index, fontsize=13)
        ax.set_ylabel("平均迁移 C-index", fontweight="bold")
        ax.set_title("跨癌种迁移：QUBO vs FullModel (Fusion)", fontweight="bold")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6)
        ax.set_ylim(0.4, max(0.65, means["mean"].max() + 0.08))
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, "P1_transfer_compare.png"))
        plt.close()

    # 3. 迁移网络图
    fusion = df[(df["ModelType"] == "Fusion") & (df["FeatureSet"] == "QUBO_Full")]
    if not fusion.empty:
        fig, ax = plt.subplots(figsize=(10, 10))
        cancers = sorted(set(fusion["TrainCancer"]) | set(fusion["TestCancer"]))
        n = len(cancers)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {c: (np.cos(a), np.sin(a)) for c, a in zip(cancers, angles)}

        for _, row in fusion.iterrows():
            s, d = row["TrainCancer"], row["TestCancer"]
            if s not in pos or d not in pos:
                continue
            x0, y0 = pos[s]; x1, y1 = pos[d]
            ci = row["C_index"]
            if ci >= 0.55:
                color, alpha, lw = "#27AE60", min(1, (ci - 0.5) * 4), 2.5
            else:
                color, alpha, lw = "#BDC3C7", 0.25, 1
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=color, alpha=alpha, lw=lw))

        for c, (x, y) in pos.items():
            ax.scatter(x, y, s=1400, color="#2C3E50", edgecolor="white", linewidth=3, zorder=10)
            ax.text(x, y, c, ha="center", va="center", fontsize=12,
                    fontweight="bold", color="white", zorder=11)

        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("跨癌种迁移网络\n(QUBO Fusion, 绿色=C-index≥0.55)",
                     fontsize=15, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, "P1_network.png"))
        plt.close()

    print("  ✅ 问题一图完成")


# ========== 问题二 ==========
def viz_problem2():
    df = load(os.path.join(P2_DIR, "within_cancer_cross_dataset_results.csv"))
    if df is None:
        print("⚠️ 问题二数据缺失，跳过")
        return

    print("问题二可视化...")
    fusion = df[df["ModelType"] == "Fusion"]
    cancers = sorted(fusion["Cancer"].unique())

    if cancers:
        ncol = 3
        nrow = int(np.ceil(len(cancers) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow))
        axes = np.array(axes).reshape(-1)

        for ax, cancer in zip(axes, cancers):
            sub = fusion[(fusion["Cancer"] == cancer) & (fusion["FeatureSet"] == "QUBO_Full")]
            if sub.empty:
                ax.axis("off")
                continue
            pivot = sub.pivot_table(index="TrainDataset", columns="TestDataset",
                                    values="C_index", aggfunc="mean")
            if SNS:
                sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                            vmin=0.45, vmax=0.75, ax=ax, cbar=False,
                            linewidths=0.5, linecolor="white")
            else:
                ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.45, vmax=0.75)
            ax.set_title(cancer, fontsize=14, fontweight="bold")
            ax.set_xlabel("Test cohort", fontsize=9)
            ax.set_ylabel("Train cohort", fontsize=9)
            ax.tick_params(labelsize=8)

        for ax in axes[len(cancers):]:
            ax.axis("off")

        fig.suptitle("同癌种跨数据集迁移 (QUBO Fusion)", fontsize=18, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, "P2_heatmap_grid.png"), bbox_inches="tight")
        plt.close()

    # 稳健性箱线图
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_data, labels = [], []
    for cancer in cancers:
        for fset in ["QUBO_Full", "FullModel"]:
            vals = fusion[(fusion["Cancer"] == cancer) & (fusion["FeatureSet"] == fset)]["C_index"].dropna()
            if len(vals) > 0:
                plot_data.append(vals.values)
                labels.append(f"{cancer}\n{fset.replace('_Full','')}")

    if plot_data:
        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(C_QUBO if "QUBO" in labels[i] else C_FULL)
            box.set_alpha(0.7)
        ax.set_ylabel("跨数据集 C-index", fontweight="bold")
        ax.set_title("同癌种跨数据集稳健性", fontweight="bold")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6)
        ax.tick_params(axis="x", labelsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, "P2_robustness_box.png"))
    plt.close()

    print("  ✅ 问题二图完成")


# ========== 问题三 ==========
def viz_problem3():
    df = load(os.path.join(P3_DIR, "qubo_vs_fullmodel_results.csv"))
    if df is None:
        print("⚠️ 问题三数据缺失，跳过")
        return

    print("问题三可视化...")
    df["BaseModel"] = df["BaseModel"].replace({"RSF-BPCox-Fusion": "Fusion"})

    mtypes = sorted(df["BaseModel"].dropna().unique())
    fig, axes = plt.subplots(1, len(mtypes), figsize=(7 * len(mtypes), 6), sharey=True)
    if len(mtypes) == 1:
        axes = [axes]

    for ax, mt in zip(axes, mtypes):
        sub = df[df["BaseModel"] == mt]
        q = sub[sub["FeatureSet"] == "QUBO_Full"].groupby("Cancer")["C_index"].mean()
        f = sub[sub["FeatureSet"] == "FullModel"].groupby("Cancer")["C_index"].mean()
        common = sorted(set(q.index) & set(f.index))
        if not common:
            ax.axis("off")
            continue
        x = np.arange(len(common))
        w = 0.38
        ax.bar(x - w/2, [q[c] for c in common], w, label="QUBO(~20基因)", color=C_QUBO, alpha=0.85)
        ax.bar(x + w/2, [f[c] for c in common], w, label="FullModel(~500基因)", color=C_FULL, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(common, rotation=30)
        ax.set_title(mt, fontweight="bold")
        ax.axhline(0.6, color="gray", linestyle="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)
        if mt == mtypes[0]:
            ax.set_ylabel("Test C-index", fontweight="bold")
            ax.legend(fontsize=9)

    fig.suptitle("QUBO 预筛 vs FullModel", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "P3_compare_grid.png"), bbox_inches="tight")
    plt.close()

    # delta 图
    fusion = df[df["BaseModel"] == "Fusion"]
    delta_rows = []
    for c in sorted(fusion["Cancer"].unique()):
        q = fusion[(fusion["Cancer"] == c) & (fusion["FeatureSet"] == "QUBO_Full")]["C_index"].mean()
        f = fusion[(fusion["Cancer"] == c) & (fusion["FeatureSet"] == "FullModel")]["C_index"].mean()
        if not np.isnan(q) and not np.isnan(f):
            delta_rows.append({"Cancer": c, "Delta": q - f})

    if delta_rows:
        dd = pd.DataFrame(delta_rows).sort_values("Delta", ascending=False)
        fig, ax = plt.subplots(figsize=(11, 6))
        colors = [C_QUBO if v >= 0 else C_FULL for v in dd["Delta"]]
        ax.bar(dd["Cancer"], dd["Delta"], color=colors, alpha=0.85)
        for i, v in enumerate(dd["Delta"]):
            ax.text(i, v + (0.003 if v >= 0 else -0.003), f"{v:+.3f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontweight="bold")
        ax.axhline(0, color="black", lw=1)
        ax.set_ylabel("Delta C-index (QUBO - FullModel)", fontweight="bold")
        ax.set_title("QUBO(~20基因) vs FullModel(~500基因) 差值（Fusion）", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, "P3_delta.png"))
        plt.close()

    print("  ✅ 问题三图完成")


def main():
    print("=" * 80)
    print("增强可视化")
    print("=" * 80)
    viz_problem1()
    viz_problem2()
    viz_problem3()
    print("\n" + "=" * 80)
    print("全部完成，图保存在:", VIZ_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
