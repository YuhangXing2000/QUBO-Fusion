# -*- coding: utf-8 -*-
"""
跨癌种分析脚本
直接读取方向二已跑完的 6 个癌种结果，不重新训练模型。

输入：
results/main（或你实际的 core_qubo_fusion.py 输出目录）

输出：
results/main/pan_cancer_analysis（或对应目录）
"""

import os
import argparse
import params
import warnings
import traceback

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except Exception:
    SEABORN_AVAILABLE = False


# =========================================================
# 0. 配置
# =========================================================

OUTPUT_ROOT = params.OUTPUT_MAIN


def parse_args(argv=None):
    """Parse CLI arguments; falls back to defaults above when none are given."""
    parser = argparse.ArgumentParser(description="QUBO-Fusion pan-cancer summary analysis")
    parser.add_argument("--results_root", dest="output_root", default=OUTPUT_ROOT,
                        help="Output root produced by core_qubo_fusion.py")
    return parser.parse_args(argv)


def apply_config(args):
    global OUTPUT_ROOT, PAN_DIR
    OUTPUT_ROOT = args.output_root
    PAN_DIR = os.path.join(OUTPUT_ROOT, "pan_cancer_analysis")


_args = parse_args()
apply_config(_args)


CANCERS = params.CANCERS

# 用于 QUBO full / no_redundancy 两个变体
QUBO_TAGS = ["qubo_full", "qubo_no_redundancy"]

# 泛癌种输出子目录
PAN_DIR = os.path.join(OUTPUT_ROOT, "pan_cancer_analysis")


# =========================================================
# 1. 工具函数
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_to_csv(df, path, **kwargs):
    ensure_dir(os.path.dirname(path))
    try:
        df.to_csv(path, **kwargs)
        return path
    except PermissionError:
        import time
        root, ext = os.path.splitext(path)
        alt = f"{root}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
        print(f"⚠️ 文件被占用，改存为: {alt}")
        df.to_csv(alt, **kwargs)
        return alt


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
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]


def setup_plot_style():
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 300
    _setup_cjk_font()
    plt.rcParams["axes.unicode_minus"] = False
    if SEABORN_AVAILABLE:
        sns.set_theme(style="whitegrid", context="talk")


setup_plot_style()
ensure_dir(PAN_DIR)

print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("PAN_DIR:", PAN_DIR)
print("CANCERS:", CANCERS)


# =========================================================
# 2. 汇总模型结果
# =========================================================

model_tables = []

for cancer in CANCERS:
    path = os.path.join(OUTPUT_ROOT, cancer, f"{cancer}_model_results.csv")

    if not os.path.exists(path):
        print(f"⚠️ 缺少模型结果: {path}")
        continue

    df = pd.read_csv(path)
    df.insert(0, "Cancer", cancer)
    model_tables.append(df)

if not model_tables:
    raise RuntimeError("没有找到任何模型结果文件")

global_df = pd.concat(model_tables, axis=0, ignore_index=True)

safe_to_csv(
    global_df,
    os.path.join(PAN_DIR, "pan_cancer_all_model_results.csv"),
    index=False,
    encoding="utf-8-sig"
)

print(f"\n已汇总 {len(global_df)} 条模型结果")


# =========================================================
# 3. 每个癌种最佳模型
# =========================================================

best_rows = []

for cancer, sub in global_df.groupby("Cancer"):
    sub = sub.sort_values("C_index", ascending=False)
    best_rows.append(sub.iloc[0].to_dict())

best_df = pd.DataFrame(best_rows)

safe_to_csv(
    best_df,
    os.path.join(PAN_DIR, "pan_cancer_best_model_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n各癌种最佳模型:")
print(best_df[["Cancer", "Model", "C_index", "logrank_p", "HR_high_vs_low"]].to_string(index=False))


# =========================================================
# 4. 主模型总结
# =========================================================

main_model = "QUBO_Full-RSF-BPCox-Fusion"
main_df = global_df[global_df["Model"] == main_model].copy()

if main_df.empty:
    print(f"\n⚠️ 没有模型 '{main_model}'，使用每个癌种第一个模型作为主模型")
    main_df = global_df.groupby("Cancer").first().reset_index()

safe_to_csv(
    main_df,
    os.path.join(PAN_DIR, "pan_cancer_main_model_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

print(f"\n主模型 '{main_model}' 跨癌种表现:")
print(main_df[["Cancer", "C_index", "logrank_p", "HR_high_vs_low"]].to_string(index=False))


# =========================================================
# 5. 跨癌种 C-index Heatmap
# =========================================================

pivot = global_df.pivot_table(
    index="Cancer",
    columns="Model",
    values="C_index",
    aggfunc="mean"
)

plt.figure(figsize=(18, max(5, pivot.shape[0] * 0.9)))

if SEABORN_AVAILABLE:
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0.5, vmax=0.75)
else:
    plt.imshow(pivot.values, cmap="YlOrRd", vmin=0.5, vmax=0.75)
    plt.colorbar()
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
    plt.yticks(range(len(pivot.index)), pivot.index)

plt.title("Pan-cancer C-index Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PAN_DIR, "pan_cancer_cindex_heatmap.png"), dpi=300)
plt.close()

print("\nC-index heatmap 已保存")


# =========================================================
# 6. 最佳模型 C-index 柱状图
# =========================================================

plt.figure(figsize=(10, 6))
bars = plt.bar(best_df["Cancer"], best_df["C_index"], color="#4C72B0")

# 在每个柱子上标注模型名
for bar, model in zip(bars, best_df["Model"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.004,
        model,
        ha="center",
        va="bottom",
        fontsize=7,
        rotation=45
    )

plt.ylim(0.5, max(0.75, best_df["C_index"].max() + 0.05))
plt.ylabel("Best test C-index")
plt.title("Best Model C-index across Cancers")
plt.tight_layout()
plt.savefig(os.path.join(PAN_DIR, "pan_cancer_best_cindex_barplot.png"), dpi=300)
plt.close()

print("最佳模型柱状图已保存")


# =========================================================
# 7. 主模型 C-index / HR / -log10(p) 柱状图
# =========================================================

# C-index
plt.figure(figsize=(10, 6))
plt.bar(main_df["Cancer"], main_df["C_index"], color="#DD8452")
plt.ylim(0.5, max(0.75, main_df["C_index"].max() + 0.03))
plt.ylabel("C-index")
plt.title(f"Main Model C-index: {main_model}")
plt.tight_layout()
plt.savefig(os.path.join(PAN_DIR, "pan_cancer_main_model_cindex_barplot.png"), dpi=300)
plt.close()

# HR
plt.figure(figsize=(10, 6))
plt.bar(main_df["Cancer"], main_df["HR_high_vs_low"], color="#55A868")
plt.axhline(1.0, color="black", linestyle="--")
plt.ylabel("HR high vs low")
plt.title(f"Main Model HR across Cancers: {main_model}")
plt.tight_layout()
plt.savefig(os.path.join(PAN_DIR, "pan_cancer_main_model_hr_barplot.png"), dpi=300)
plt.close()

# -log10(p)
temp = main_df.copy()
temp["minus_log10_p"] = -np.log10(temp["logrank_p"].clip(lower=1e-300))

plt.figure(figsize=(10, 6))
plt.bar(temp["Cancer"], temp["minus_log10_p"], color="#C44E52")
plt.axhline(-np.log10(0.05), color="black", linestyle="--", label="p=0.05")
plt.ylabel("-log10(log-rank p)")
plt.title(f"Main Model Log-rank Significance: {main_model}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PAN_DIR, "pan_cancer_main_model_logrank_barplot.png"), dpi=300)
plt.close()

print("主模型三图已保存")


# =========================================================
# 8. 消融：QUBO_Full vs RandomGenes vs TopCoxGenes 汇总
# =========================================================

ablation_models = [
    ("QUBO_Full-RSF-BPCox-Fusion", "QUBO + Fusion"),
    ("RandomGenes-RSF-BPCox-Fusion", "Random + Fusion"),
    ("TopCoxGenes-RSF-BPCox-Fusion", "TopCox + Fusion"),
]

ablation_rows = []

for model_name, model_label in ablation_models:
    sub = global_df[global_df["Model"] == model_name].copy()

    if sub.empty:
        continue

    for _, row in sub.iterrows():
        ablation_rows.append({
            "Cancer": row["Cancer"],
            "Model": model_label,
            "C_index": row["C_index"],
            "logrank_p": row["logrank_p"],
            "HR_high_vs_low": row["HR_high_vs_low"],
        })

if ablation_rows:
    ablation_df = pd.DataFrame(ablation_rows)

    safe_to_csv(
        ablation_df,
        os.path.join(PAN_DIR, "pan_cancer_ablation_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 分组柱状图
    cancers = sorted(ablation_df["Cancer"].unique())
    model_labels = [m[1] for m in ablation_models if ablation_df["Model"].str.contains(m[1]).any()]

    if model_labels:
        x = np.arange(len(cancers))
        width = 0.25
        colors = ["#DD8452", "#4C72B0", "#55A868"]

        plt.figure(figsize=(12, 6))

        for i, (mlabel, color) in enumerate(zip(model_labels, colors[:len(model_labels)])):
            vals = []
            for cancer in cancers:
                sub = ablation_df[(ablation_df["Cancer"] == cancer) & (ablation_df["Model"] == mlabel)]
                vals.append(sub["C_index"].values[0] if not sub.empty else np.nan)
            plt.bar(x + i * width, vals, width, label=mlabel, color=color)

        plt.xticks(x + width, cancers)
        plt.ylabel("C-index")
        plt.title("Ablation: QUBO vs Random vs TopCox (Fusion Model)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PAN_DIR, "pan_cancer_ablation_fusion_cindex.png"), dpi=300)
        plt.close()

        print("消融对比图已保存")


# =========================================================
# 9. 汇总 QUBO 选择基因
# =========================================================

selected_gene_records = []

for cancer in CANCERS:
    for tag in QUBO_TAGS:
        path = os.path.join(OUTPUT_ROOT, cancer, f"{tag}_selected_genes.csv")

        if not os.path.exists(path):
            print(f"⚠️ 缺少选择基因文件: {path}")
            continue

        df = pd.read_csv(path)

        # 兼容不同列名
        gene_col = "gene" if "gene" in df.columns else df.columns[0]

        for _, row in df.iterrows():
            selected_gene_records.append({
                "Cancer": cancer,
                "Method": tag,
                "Gene": str(row[gene_col]),
                "rank_score": float(row["rank_score"]) if "rank_score" in row else np.nan,
                "cindex": float(row["cindex"]) if "cindex" in row else np.nan,
                "p": float(row["p"]) if "p" in row else np.nan,
            })

if not selected_gene_records:
    print("⚠️ 未找到 QUBO 选择基因，跳过基因分析")
    import sys
    sys.exit(0)

gene_df = pd.DataFrame(selected_gene_records)

safe_to_csv(
    gene_df,
    os.path.join(PAN_DIR, "pan_cancer_all_selected_genes.csv"),
    index=False,
    encoding="utf-8-sig"
)

print(f"\n已汇总 {len(gene_df)} 条选择基因记录")


# =========================================================
# 10. 泛癌种基因频率
# =========================================================

for method in QUBO_TAGS:
    sub = gene_df[gene_df["Method"] == method].copy()

    if sub.empty:
        continue

    freq_rows = []

    for gene, gsub in sub.groupby("Gene"):
        cancer_list = sorted(gsub["Cancer"].unique().tolist())
        freq_rows.append({
            "Gene": gene,
            "CancerCount": len(cancer_list),
            "Cancers": ";".join(cancer_list),
            "MeanRankScore": float(pd.to_numeric(gsub["rank_score"], errors="coerce").mean())
            if "rank_score" in gsub else np.nan,
            "MeanCindex": float(pd.to_numeric(gsub["cindex"], errors="coerce").mean())
            if "cindex" in gsub else np.nan,
        })

    freq_df = pd.DataFrame(freq_rows)
    freq_df = freq_df.sort_values(["CancerCount", "MeanRankScore"], ascending=[False, False])

    safe_to_csv(
        freq_df,
        os.path.join(PAN_DIR, f"pan_cancer_gene_frequency_{method}.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 频率柱状图 top 30
    top = freq_df.head(30)

    plt.figure(figsize=(12, 7))
    colors = ["#D62728" if c >= 3 else "#4C72B0" for c in top["CancerCount"]]
    plt.bar(top["Gene"], top["CancerCount"], color=colors)
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Number of cancers")
    plt.title(f"Pan-cancer recurrent selected genes: {method}")
    plt.tight_layout()
    plt.savefig(os.path.join(PAN_DIR, f"pan_cancer_gene_frequency_{method}.png"), dpi=300)
    plt.close()

    # 基因 × 癌种 binary 矩阵
    genes = freq_df["Gene"].tolist()
    mat = pd.DataFrame(0, index=genes, columns=CANCERS)

    for _, row in sub.iterrows():
        mat.loc[row["Gene"], row["Cancer"]] = 1

    safe_to_csv(
        mat,
        os.path.join(PAN_DIR, f"pan_cancer_gene_binary_matrix_{method}.csv"),
        encoding="utf-8-sig"
    )

    # 只画出现 ≥ 2 癌种的基因
    mat2 = mat.loc[mat.sum(axis=1) >= 2]

    if not mat2.empty:
        plt.figure(figsize=(8, max(4, mat2.shape[0] * 0.35)))

        if SEABORN_AVAILABLE:
            sns.heatmap(mat2, cmap="Blues", cbar=False, linewidths=0.5, linecolor="gray")
        else:
            plt.imshow(mat2.values, cmap="Blues", aspect="auto")
            plt.xticks(range(len(mat2.columns)), mat2.columns)
            plt.yticks(range(len(mat2.index)), mat2.index)

        plt.title(f"Recurrent selected genes across cancers: {method}")
        plt.tight_layout()
        plt.savefig(os.path.join(PAN_DIR, f"pan_cancer_recurrent_gene_heatmap_{method}.png"), dpi=300)
        plt.close()

    print(f"\n{method}:")
    print(f"  共 {len(freq_df)} 个基因")
    print(f"  出现在 2+ 癌种: {(freq_df['CancerCount'] >= 2).sum()} 个")
    print(f"  出现在 3+ 癌种: {(freq_df['CancerCount'] >= 3).sum()} 个")
    if not freq_df.empty:
        print(f"  最高频率基因: {', '.join(freq_df.head(10)['Gene'].tolist())}")


# =========================================================
# 11. 癌种间 Jaccard 相似度
# =========================================================

for method in QUBO_TAGS:
    sub = gene_df[gene_df["Method"] == method].copy()

    if sub.empty:
        continue

    gene_sets = {}

    for cancer in CANCERS:
        gene_sets[cancer] = set(sub[sub["Cancer"] == cancer]["Gene"].astype(str).tolist())

    jac = pd.DataFrame(index=CANCERS, columns=CANCERS, dtype=float)

    for c1 in CANCERS:
        for c2 in CANCERS:
            s1 = gene_sets.get(c1, set())
            s2 = gene_sets.get(c2, set())
            union = len(s1 | s2)
            jac.loc[c1, c2] = len(s1 & s2) / union if union > 0 else np.nan

    safe_to_csv(
        jac,
        os.path.join(PAN_DIR, f"pan_cancer_gene_jaccard_{method}.csv"),
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(8, 7))

    if SEABORN_AVAILABLE:
        sns.heatmap(jac, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)
    else:
        plt.imshow(jac.values.astype(float), cmap="YlGnBu", vmin=0, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(CANCERS)), CANCERS)
        plt.yticks(range(len(CANCERS)), CANCERS)

    plt.title(f"Jaccard similarity of selected gene sets: {method}")
    plt.tight_layout()
    plt.savefig(os.path.join(PAN_DIR, f"pan_cancer_gene_jaccard_heatmap_{method}.png"), dpi=300)
    plt.close()

    print(f"\n{method} Jaccard 相似度:")
    print(jac.to_string())


# =========================================================
# 12. Feature 消融 + 模型消融综合热图
# =========================================================

summary_pivot = global_df.pivot_table(
    index=["Cancer", "FeatureSet"],
    columns="Model",
    values="C_index",
    aggfunc="mean"
)

if not summary_pivot.empty:
    safe_to_csv(
        summary_pivot.reset_index(),
        os.path.join(PAN_DIR, "pan_cancer_feature_model_cindex_table.csv"),
        index=False,
        encoding="utf-8-sig"
    )

print("\n" + "=" * 100)
print("跨癌种分析完成")
print("输出目录:", PAN_DIR)
print("=" * 100)
