# -*- coding: utf-8 -*-
"""
三个问题完整分析_增强版.py

独立完成三个问题，不修改原始 core_qubo_fusion.py。
增加精美可视化、完整统计、扩展分析。

问题 1：跨癌种迁移验证
问题 2：同一癌种内跨 dataset/cohort 验证
问题 3：QUBO 预筛 + ML vs 未筛选 FullModel 对比
"""

import os
import json
import time
import argparse
import params
import warnings
import traceback
import importlib.util

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except Exception:
    SEABORN_AVAILABLE = False

# =========================================================
# 0. 配置
# =========================================================

CANCER2_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core_qubo_fusion.py")
DATA_ROOT = params.DATA_ROOT
RESULTS_ROOT = params.OUTPUT_MAIN
OUTPUT_ROOT = params.OUTPUT_EXTENSIONS


def parse_args(argv=None):
    """Parse CLI arguments; falls back to defaults above when none are given."""
    parser = argparse.ArgumentParser(description="QUBO-Fusion three extension analyses")
    parser.add_argument("--cancer2_script", default=CANCER2_SCRIPT,
                        help="Path to core_qubo_fusion.py (provides shared model functions)")
    parser.add_argument("--data_root", default=DATA_ROOT,
                        help="Root dir of raw data (mrna.rda files, one subdir per cancer)")
    parser.add_argument("--results_root", default=RESULTS_ROOT,
                        help="Output root produced by core_qubo_fusion.py")
    parser.add_argument("--output", dest="output_root", default=OUTPUT_ROOT,
                        help="Output root for extension analysis results")
    return parser.parse_args(argv)


def apply_config(args):
    global CANCER2_SCRIPT, DATA_ROOT, RESULTS_ROOT, OUTPUT_ROOT
    CANCER2_SCRIPT = args.cancer2_script
    DATA_ROOT = args.data_root
    RESULTS_ROOT = args.results_root
    OUTPUT_ROOT = args.output_root


_args = parse_args()
apply_config(_args)


CANCERS = params.CANCERS
QUBO_TAG = "qubo_full"

RANDOM_SEED = params.RANDOM_SEED

MIN_CROSS_TRAIN_SAMPLE = params.MIN_CROSS_TRAIN_SAMPLE
MIN_CROSS_TEST_SAMPLE = params.MIN_CROSS_TEST_SAMPLE
MIN_CROSS_EVENT = params.MIN_CROSS_EVENT
MIN_CROSS_GENE = params.MIN_CROSS_GENE

MIN_TRANSFER_TRAIN_SAMPLE = params.MIN_TRANSFER_TRAIN_SAMPLE
MIN_TRANSFER_TEST_SAMPLE = params.MIN_TRANSFER_TEST_SAMPLE
MIN_TRANSFER_EVENT = params.MIN_TRANSFER_EVENT
MIN_TRANSFER_GENE = params.MIN_TRANSFER_GENE

FULL_MODEL_GENES = params.FULL_MODEL_GENES
HIGH_DIM_COX_LIMIT = params.HIGH_DIM_COX_LIMIT

# 颜色方案
COLOR_QUBO = "#E74C3C"
COLOR_FULL = "#3498DB"
COLOR_RANDOM = "#95A5A6"
COLOR_TOP = "#F39C12"

COLORS_MODEL = {
    "RSF": "#E74C3C",
    "BP-Cox": "#3498DB",
    "Fusion": "#2ECC71"
}


# =========================================================
# 1. 导入 癌症2.py
# =========================================================

def import_cancer2(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到: {path}")

    spec = importlib.util.spec_from_file_location("cancer2_base", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d2 = import_cancer2(CANCER2_SCRIPT)
d2.DATA_ROOT = DATA_ROOT
d2.OUTPUT_ROOT = RESULTS_ROOT


# =========================================================
# 2. 基础工具
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_csv(df, path):
    ensure_dir(os.path.dirname(path))
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  ✅ 已保存: {os.path.basename(path)}")
    except PermissionError:
        alt = path.replace(".csv", f"_{time.strftime('%H%M%S')}.csv")
        df.to_csv(alt, index=False, encoding="utf-8-sig")
        print(f"  ✅ 已保存(替代): {os.path.basename(alt)}")


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


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


def setup_plots():
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 300
    _setup_cjk_font()
    plt.rcParams["axes.unicode_minus"] = False

    if SEABORN_AVAILABLE:
        sns.set_theme(style="whitegrid", context="talk", palette="muted")
        sns.set_context("talk", rc={"lines.linewidth": 2.5})


setup_plots()
ensure_dir(OUTPUT_ROOT)


def read_qubo_genes(cancer, tag="qubo_full"):
    path = os.path.join(RESULTS_ROOT, cancer, f"{tag}_selected_genes.csv")

    if not os.path.exists(path):
        return []

    df = pd.read_csv(path)
    gene_col = "gene" if "gene" in df.columns else df.columns[0]
    genes = df[gene_col].dropna().astype(str).tolist()

    return [g.strip() for g in genes if g.strip()]


def prepare_data(expr, surv, genes):
    genes = [g for g in genes if g in expr.columns]
    X = expr[genes].values

    imp = d2.SimpleImputer(strategy="median")
    sc = d2.StandardScaler()

    X = sc.fit_transform(imp.fit_transform(X))
    return X, genes


# =========================================================
# 3. 模型训练封装
# =========================================================

def train_rsf(X_train, surv_train, X_test):
    params = {"n_estimators": 80, "max_depth": 8, "min_samples_leaf": 5}

    try:
        best = d2.cv_tune_rsf(X_train, surv_train)
        if best is not None:
            params.update(best)
    except Exception:
        pass

    _, rsf_tr, rsf_te = d2.fit_rsf_predict(X_train, surv_train, X_test, params)
    return rsf_tr, rsf_te


def train_bpcox(X_train, surv_train, X_test):
    params = {
        "hidden_dim": 32,
        "dropout": 0.20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 50
    }

    try:
        best = d2.cv_tune_bpcox(X_train, surv_train)
        if best is not None:
            params.update(best)
    except Exception:
        pass

    _, bp_tr, bp_te = d2.fit_bpcox_predict(X_train, surv_train, X_test, params)
    return bp_tr, bp_te


def train_fusion(X_train, surv_train, X_test, cv_folds=2):
    folds = d2.make_cv_splits(surv_train, cv_folds)

    fold_cache = []

    for tr_idx, va_idx in folds:
        try:
            rsf_tr, rsf_va = train_rsf(X_train[tr_idx], surv_train.iloc[tr_idx], X_train[va_idx])
            bp_tr, bp_va = train_bpcox(X_train[tr_idx], surv_train.iloc[tr_idx], X_train[va_idx])

            _, rsf_va_s = d2.standardize_risk_by_train(rsf_tr, rsf_va)
            _, bp_va_s = d2.standardize_risk_by_train(bp_tr, bp_va)

            fold_cache.append({
                "val_index": va_idx,
                "rsf_va": rsf_va_s,
                "bp_va": bp_va_s
            })

        except Exception:
            traceback.print_exc()

    best_w, best_score = 0.5, -np.inf

    for w in np.linspace(0, 1, 6):
        vals = []

        for item in fold_cache:
            risk = w * item["rsf_va"] + (1 - w) * item["bp_va"]

            try:
                vals.append(d2.calc_cindex(surv_train.iloc[item["val_index"]], risk))
            except Exception:
                pass

        score = np.mean(vals) if vals else -np.inf

        if score > best_score:
            best_w, best_score = float(w), score

    rsf_tr, rsf_te = train_rsf(X_train, surv_train, X_test)
    bp_tr, bp_te = train_bpcox(X_train, surv_train, X_test)

    _, rsf_te_s = d2.standardize_risk_by_train(rsf_tr, rsf_te)
    _, bp_te_s = d2.standardize_risk_by_train(bp_tr, bp_te)

    fusion = best_w * rsf_te_s + (1 - best_w) * bp_te_s

    return rsf_te, bp_te, fusion, best_w, best_score


def evaluate_three_models(X_train, surv_train, X_test, surv_test, prefix):
    rows = []

    # RSF
    try:
        print(f"      训练 {prefix}-RSF ...")
        _, risk = train_rsf(X_train, surv_train, X_test)

        r = d2.evaluate_survival_model(f"{prefix}-RSF", surv_test, risk)
        r.update({"ModelType": "RSF", "FeatureNum": X_train.shape[1]})
        rows.append(r)
    except Exception:
        traceback.print_exc()

    # BP-Cox
    try:
        print(f"      训练 {prefix}-BP-Cox ...")
        _, risk = train_bpcox(X_train, surv_train, X_test)

        r = d2.evaluate_survival_model(f"{prefix}-BP-Cox", surv_test, risk)
        r.update({"ModelType": "BP-Cox", "FeatureNum": X_train.shape[1]})
        rows.append(r)
    except Exception:
        traceback.print_exc()

    # Fusion
    try:
        print(f"      训练 {prefix}-Fusion ...")
        _, _, risk, w, w_score = train_fusion(X_train, surv_train, X_test)

        r = d2.evaluate_survival_model(f"{prefix}-Fusion", surv_test, risk)
        r.update({
            "ModelType": "Fusion",
            "FeatureNum": X_train.shape[1],
            "CV_C_index": w_score,
            "FusionWeight_RSF": w,
            "FusionWeight_BPCox": 1 - w
        })
        rows.append(r)
    except Exception:
        traceback.print_exc()

    return rows


# =========================================================
# 4. 问题 1：跨癌种迁移验证
# =========================================================

def cross_cancer_transfer():
    print("\n" + "=" * 100)
    print("问题 1：跨癌种迁移验证 - Cancer A biomarker/model -> Cancer B 预后意义")
    print("=" * 100)

    out_dir = os.path.join(OUTPUT_ROOT, "01_cross_cancer_transfer")
    ensure_dir(out_dir)

    print("\n加载数据 ...")
    data = {}

    for cancer in CANCERS:
        try:
            expr, surv, info = d2.load_cancer_data(cancer)

            if expr.shape[0] >= MIN_TRANSFER_TRAIN_SAMPLE and surv["event"].sum() >= MIN_TRANSFER_EVENT:
                data[cancer] = {"expr": expr, "surv": surv}
                print(f"  ✅ {cancer}: n={expr.shape[0]}, genes={expr.shape[1]}, events={int(surv['event'].sum())}")

        except Exception:
            traceback.print_exc()

    if len(data) < 2:
        print("⚠️ 有效癌种 < 2，跳过")
        return None

    all_rows = []

    for train_cancer in CANCERS:
        if train_cancer not in data:
            continue

        qubo_genes = read_qubo_genes(train_cancer, QUBO_TAG)

        if len(qubo_genes) == 0:
            print(f"  ⚠️ {train_cancer} 无 QUBO 基因")
            continue

        train_expr = data[train_cancer]["expr"]
        train_surv = data[train_cancer]["surv"]

        for test_cancer in CANCERS:
            if test_cancer == train_cancer or test_cancer not in data:
                continue

            test_expr = data[test_cancer]["expr"]
            test_surv = data[test_cancer]["surv"]

            common_all = sorted(set(train_expr.columns) & set(test_expr.columns))

            if len(common_all) < MIN_TRANSFER_GENE:
                continue

            print(f"\n  🔄 {train_cancer} -> {test_cancer}")

            qubo_common = [g for g in qubo_genes if g in common_all]

            if len(common_all) > FULL_MODEL_GENES:
                var = train_expr[common_all].var().sort_values(ascending=False)
                full_common = var.index[:FULL_MODEL_GENES].tolist()
            else:
                full_common = common_all

            feature_sets = {}

            if len(qubo_common) >= MIN_TRANSFER_GENE:
                feature_sets["QUBO_Full"] = qubo_common

            if len(full_common) >= MIN_TRANSFER_GENE:
                feature_sets["FullModel"] = full_common

            for fname, genes in feature_sets.items():
                print(f"    FeatureSet={fname}, genes={len(genes)}")

                try:
                    X_tr, _ = prepare_data(train_expr, train_surv, genes)
                    X_te, _ = prepare_data(test_expr, test_surv, genes)

                    rows = evaluate_three_models(
                        X_tr, train_surv,
                        X_te, test_surv,
                        fname
                    )

                    for r in rows:
                        r.update({
                            "TrainCancer": train_cancer,
                            "TestCancer": test_cancer,
                            "FeatureSet": fname,
                            "TrainSampleNum": int(train_expr.shape[0]),
                            "TestSampleNum": int(test_expr.shape[0]),
                            "TrainEventNum": int(train_surv["event"].sum()),
                            "TestEventNum": int(test_surv["event"].sum()),
                            "CommonGeneNum": len(common_all),
                        })

                        all_rows.append(r)

                except Exception:
                    traceback.print_exc()

    if not all_rows:
        print("⚠️ 无有效结果")
        return None

    df = pd.DataFrame(all_rows)
    safe_csv(df, os.path.join(out_dir, "cross_cancer_transfer_results.csv"))

    # 可视化
    plot_cross_cancer_heatmaps(df, out_dir)
    plot_cross_cancer_summary(df, out_dir)
    plot_cross_cancer_network(df, out_dir)

    print(f"\n✅ 问题 1 完成：{df.shape[0]} 条结果")
    return df


def plot_cross_cancer_heatmaps(df, out_dir):
    """跨癌种 C-index 热图"""
    for fset in sorted(df["FeatureSet"].unique()):
        for mtype in sorted(df["ModelType"].unique()):
            sub = df[(df["FeatureSet"] == fset) & (df["ModelType"] == mtype)]

            if sub.empty:
                continue

            pivot = sub.pivot_table(
                index="TrainCancer",
                columns="TestCancer",
                values="C_index",
                aggfunc="mean"
            )

            safe_csv(pivot, os.path.join(out_dir, f"cindex_matrix_{fset}_{mtype}.csv"))

            fig, ax = plt.subplots(figsize=(8, 6))

            if SEABORN_AVAILABLE:
                sns.heatmap(
                    pivot,
                    annot=True,
                    fmt=".3f",
                    cmap="RdYlGn",
                    vmin=0.5,
                    vmax=0.8,
                    ax=ax,
                    cbar_kws={"label": "C-index"}
                )
            else:
                im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.5, vmax=0.8)
                plt.colorbar(im, ax=ax)
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index)

            ax.set_title(f"Cross-cancer transfer C-index\n{fset} + {mtype}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Test Cancer", fontsize=12)
            ax.set_ylabel("Train Cancer", fontsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"heatmap_{fset}_{mtype}.png"))
            plt.close()


def plot_cross_cancer_summary(df, out_dir):
    """跨癌种平均 C-index 条形图"""
    summary = df.groupby(["FeatureSet", "ModelType"])["C_index"].agg(["mean", "std", "count"]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(summary))
    width = 0.8

    colors = [COLOR_QUBO if "QUBO" in str(row["FeatureSet"]) else COLOR_FULL for _, row in summary.iterrows()]

    bars = ax.bar(x, summary["mean"], width, yerr=summary["std"], color=colors, alpha=0.8, capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['FeatureSet']}\n{row['ModelType']}" for _, row in summary.iterrows()], fontsize=9)
    ax.set_ylabel("Mean C-index", fontsize=12, fontweight="bold")
    ax.set_title("Cross-cancer transfer performance summary", fontsize=14, fontweight="bold")
    ax.set_ylim(0.5, min(0.8, summary["mean"].max() + 0.1))
    ax.axhline(0.6, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.grid(axis="y", alpha=0.3)

    # 图例
    qubo_patch = mpatches.Patch(color=COLOR_QUBO, label="QUBO preselected")
    full_patch = mpatches.Patch(color=COLOR_FULL, label="FullModel")
    ax.legend(handles=[qubo_patch, full_patch], loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_barplot.png"))
    plt.close()


def plot_cross_cancer_network(df, out_dir):
    """跨癌种网络图：有效迁移连线"""
    fusion_df = df[df["ModelType"] == "Fusion"]

    if fusion_df.empty:
        return

    # 只画 C-index > 0.6 的有效迁移
    good = fusion_df[fusion_df["C_index"] > 0.6]

    if good.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    cancers = sorted(set(good["TrainCancer"]) | set(good["TestCancer"]))
    n = len(cancers)

    # 圆形布局
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {c: (np.cos(a), np.sin(a)) for c, a in zip(cancers, angles)}

    # 画连线
    for _, row in good.iterrows():
        src = row["TrainCancer"]
        dst = row["TestCancer"]

        if src not in pos or dst not in pos:
            continue

        x0, y0 = pos[src]
        x1, y1 = pos[dst]

        color = COLOR_QUBO if row["FeatureSet"] == "QUBO_Full" else COLOR_FULL
        alpha = min(1.0, (row["C_index"] - 0.5) * 2)

        ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, linewidth=2)

    # 画节点
    for c, (x, y) in pos.items():
        ax.scatter(x, y, s=800, color="#34495E", edgecolor="white", linewidth=2, zorder=10)
        ax.text(x, y, c, ha="center", va="center", fontsize=11, fontweight="bold", color="white", zorder=11)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Cross-cancer effective transfer network\n(C-index > 0.6, Fusion model)",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    qubo_line = plt.Line2D([0], [0], color=COLOR_QUBO, linewidth=2, label="QUBO")
    full_line = plt.Line2D([0], [0], color=COLOR_FULL, linewidth=2, label="FullModel")
    ax.legend(handles=[qubo_line, full_line], loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "network_effective_transfer.png"))
    plt.close()


# =========================================================
# 5. 问题 2：同一癌种内跨 dataset/cohort
# =========================================================

def within_cancer_cross_dataset():
    print("\n" + "=" * 100)
    print("问题 2：同一癌种内跨 dataset/cohort 验证 - cohort A -> cohort B 稳健性")
    print("=" * 100)

    out_dir = os.path.join(OUTPUT_ROOT, "02_within_cancer_cross_dataset")
    ensure_dir(out_dir)

    all_rows = []

    for cancer in CANCERS:
        print(f"\n处理癌种: {cancer}")

        try:
            expr, surv, info = d2.load_cancer_data(cancer)
        except Exception:
            traceback.print_exc()
            continue

        if "cohort" not in surv.columns:
            print(f"  ⚠️ {cancer} 无 cohort 列")
            continue

        cohorts = surv.groupby("cohort")

        valid_cohorts = []

        for cohort, idx in cohorts.groups.items():
            sub_surv = surv.loc[idx]

            if sub_surv.shape[0] >= MIN_CROSS_TRAIN_SAMPLE and sub_surv["event"].sum() >= MIN_CROSS_EVENT:
                valid_cohorts.append(cohort)

        if len(valid_cohorts) < 2:
            print(f"  ⚠️ {cancer} 有效 cohort < 2")
            continue

        print(f"  有效 cohort: {len(valid_cohorts)}")

        # 每个 train cohort 跑 QUBO
        qubo_cache = {}

        for train_cohort in valid_cohorts:
            idx = cohorts.groups[train_cohort]
            tr_expr = expr.loc[idx].copy()
            tr_surv = surv.loc[idx].copy()

            if tr_expr.shape[0] < MIN_CROSS_TRAIN_SAMPLE or tr_surv["event"].sum() < MIN_CROSS_EVENT:
                continue

            try:
                score_df = d2.univariate_cox_scores(tr_expr, tr_surv, d2.MAX_UNIVARIATE_GENES)

                qubo_genes, _ = d2.select_genes_by_qubo(
                    tr_expr,
                    score_df,
                    out_dir,
                    tag=f"{cancer}_{train_cohort}_qubo",
                    use_redundancy=True,
                    redundancy_weight=d2.QUBO_REDUNDANCY_WEIGHT
                )

                qubo_cache[train_cohort] = qubo_genes

            except Exception:
                traceback.print_exc()

        # 跨 cohort 验证
        for train_cohort in valid_cohorts:
            if train_cohort not in qubo_cache:
                continue

            tr_idx = cohorts.groups[train_cohort]
            tr_expr = expr.loc[tr_idx].copy()
            tr_surv = surv.loc[tr_idx].copy()

            qubo_genes = qubo_cache[train_cohort]

            for test_cohort in valid_cohorts:
                if test_cohort == train_cohort:
                    continue

                te_idx = cohorts.groups[test_cohort]
                te_expr = expr.loc[te_idx].copy()
                te_surv = surv.loc[te_idx].copy()

                if te_expr.shape[0] < MIN_CROSS_TEST_SAMPLE or te_surv["event"].sum() < MIN_CROSS_EVENT:
                    continue

                common_all = sorted(set(tr_expr.columns) & set(te_expr.columns))

                if len(common_all) < MIN_CROSS_GENE:
                    continue

                qubo_common = [g for g in qubo_genes if g in common_all]

                if len(common_all) > FULL_MODEL_GENES:
                    var = tr_expr[common_all].var().sort_values(ascending=False)
                    full_common = var.index[:FULL_MODEL_GENES].tolist()
                else:
                    full_common = common_all

                feature_sets = {}

                if len(qubo_common) >= MIN_CROSS_GENE:
                    feature_sets["QUBO_Full"] = qubo_common

                if len(full_common) >= MIN_CROSS_GENE:
                    feature_sets["FullModel"] = full_common

                for fname, genes in feature_sets.items():
                    print(f"    {cancer}: {train_cohort} -> {test_cohort}, {fname}, genes={len(genes)}")

                    try:
                        X_tr, _ = prepare_data(tr_expr, tr_surv, genes)
                        X_te, _ = prepare_data(te_expr, te_surv, genes)

                        rows = evaluate_three_models(
                            X_tr, tr_surv,
                            X_te, te_surv,
                            fname
                        )

                        for r in rows:
                            r.update({
                                "Cancer": cancer,
                                "TrainDataset": train_cohort,
                                "TestDataset": test_cohort,
                                "FeatureSet": fname,
                            })

                            all_rows.append(r)

                    except Exception:
                        traceback.print_exc()

    if not all_rows:
        print("⚠️ 无有效结果")
        return None

    df = pd.DataFrame(all_rows)
    safe_csv(df, os.path.join(out_dir, "within_cancer_cross_dataset_results.csv"))

    # 可视化
    plot_within_cancer_summary(df, out_dir)
    plot_within_cancer_heatmaps(df, out_dir)

    print(f"\n✅ 问题 2 完成：{df.shape[0]} 条结果")
    return df


def plot_within_cancer_summary(df, out_dir):
    """同癌种跨数据集汇总条形图"""
    summary = df.groupby(["Cancer", "FeatureSet", "ModelType"])["C_index"].agg(["mean", "count"]).reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))

    cancers = sorted(summary["Cancer"].unique())
    fsets = sorted(summary["FeatureSet"].unique())
    mtypes = sorted(summary["ModelType"].unique())

    x = np.arange(len(cancers))
    width = 0.15

    for i, (fset, mtype) in enumerate([(f, m) for f in fsets for m in mtypes]):
        sub = summary[(summary["FeatureSet"] == fset) & (summary["ModelType"] == mtype)]

        if sub.empty:
            continue

        vals = [sub[sub["Cancer"] == c]["mean"].values[0] if not sub[sub["Cancer"] == c].empty else 0 for c in cancers]

        color = COLOR_QUBO if fset == "QUBO_Full" else COLOR_FULL
        alpha = 0.9 if mtype == "Fusion" else (0.7 if mtype == "RSF" else 0.5)

        ax.bar(x + i * width - width * 2, vals, width, label=f"{fset}-{mtype}", color=color, alpha=alpha)

    ax.set_xticks(x)
    ax.set_xticklabels(cancers, fontsize=10)
    ax.set_ylabel("Mean C-index", fontsize=12, fontweight="bold")
    ax.set_title("Within-cancer cross-dataset performance", fontsize=14, fontweight="bold")
    ax.set_ylim(0.5, 0.85)
    ax.axhline(0.6, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_barplot.png"))
    plt.close()

def plot_within_cancer_heatmaps(df, out_dir):
    """每个癌种的 cohort A -> cohort B C-index 热图（Fusion）"""
    fusion = df[df["ModelType"] == "Fusion"]

    if fusion.empty:
        return

    for cancer in sorted(fusion["Cancer"].unique()):
        for fset in sorted(fusion["FeatureSet"].unique()):
            sub = fusion[(fusion["Cancer"] == cancer) & (fusion["FeatureSet"] == fset)]

            if sub.empty:
                continue

            pivot = sub.pivot_table(
                index="TrainDataset",
                columns="TestDataset",
                values="C_index",
                aggfunc="mean"
            )

            if pivot.empty:
                continue

            safe_csv(
                pivot.reset_index(),
                os.path.join(out_dir, f"matrix_{cancer}_{fset}.csv")
            )

            fig, ax = plt.subplots(
                figsize=(max(6, pivot.shape[1] * 1.1), max(5, pivot.shape[0] * 0.9))
            )

            if SEABORN_AVAILABLE:
                sns.heatmap(
                    pivot,
                    annot=True,
                    fmt=".3f",
                    cmap="RdYlGn",
                    vmin=0.5,
                    vmax=0.8,
                    ax=ax,
                    cbar_kws={"label": "C-index"}
                )
            else:
                im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.5, vmax=0.8)
                plt.colorbar(im, ax=ax)
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index)

            ax.set_title(
                f"{cancer} within-cancer cross-dataset\n{fset} + Fusion",
                fontsize=13,
                fontweight="bold"
            )
            ax.set_xlabel("Test dataset / cohort", fontsize=11)
            ax.set_ylabel("Train dataset / cohort", fontsize=11)

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"heatmap_{cancer}_{fset}.png"))
            plt.close()


# =========================================================
# 6. 问题 3：QUBO 预筛 + ML vs 未筛选 FullModel
# =========================================================

def qubo_vs_fullmodel():
    print("\n" + "=" * 100)
    print("问题 3：QUBO 预筛 biomarker + ML 模型 vs 未筛选 FullModel")
    print("=" * 100)

    out_dir = os.path.join(OUTPUT_ROOT, "03_qubo_vs_fullmodel")
    ensure_dir(out_dir)

    all_rows = []

    # 方式 A：优先从 癌症2.py 已有结果汇总
    for cancer in CANCERS:
        path = os.path.join(RESULTS_ROOT, cancer, f"{cancer}_model_results.csv")

        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        if "FeatureSet" not in df.columns or "Model" not in df.columns:
            continue

        ml_models = ["RSF-BPCox-Fusion", "RSF", "BP-Cox"]

        for _, row in df.iterrows():
            model_str = str(row["Model"])
            feature_set = str(row["FeatureSet"])

            if feature_set not in ["QUBO_Full", "FullModel"]:
                continue

            base_model = None
            for m in ml_models:
                if m in model_str:
                    base_model = m
                    break

            if base_model is None:
                continue

            all_rows.append({
                "Cancer": cancer,
                "FeatureSet": feature_set,
                "BaseModel": base_model,
                "Model": model_str,
                "C_index": float(row.get("C_index", np.nan)),
                "logrank_p": float(row.get("logrank_p", np.nan)),
                "HR_high_vs_low": float(row.get("HR_high_vs_low", np.nan)),
                "FeatureNum": int(row.get("FeatureNum", 0)),
                "Source": "from_results"
            })

    # 方式 B：如果已有结果不含 FullModel，则独立训练补齐
    have_full = any(r["FeatureSet"] == "FullModel" for r in all_rows)

    if not have_full:
        print("\n已有结果不含 FullModel，改为独立训练 QUBO_Full 与 FullModel ...")

        for cancer in CANCERS:
            print(f"\n处理癌种: {cancer}")

            try:
                expr, surv, _ = d2.load_cancer_data(cancer)
            except Exception:
                traceback.print_exc()
                continue

            stratify = surv["event"] if surv["event"].value_counts().min() >= 2 else None

            expr_tr, expr_te, surv_tr, surv_te = d2.train_test_split(
                expr, surv,
                test_size=0.3,
                random_state=RANDOM_SEED,
                stratify=stratify
            )

            qubo_genes = read_qubo_genes(cancer, QUBO_TAG)

            if len(qubo_genes) == 0:
                continue

            all_genes = list(expr_tr.columns)

            if len(all_genes) > FULL_MODEL_GENES:
                var = expr_tr[all_genes].var().sort_values(ascending=False)
                full_genes = var.index[:FULL_MODEL_GENES].tolist()
            else:
                full_genes = all_genes

            for fname, genes in [("QUBO_Full", qubo_genes), ("FullModel", full_genes)]:
                print(f"  FeatureSet={fname}, genes={len(genes)}")

                try:
                    X_tr, _ = prepare_data(expr_tr, surv_tr, genes)
                    X_te, _ = prepare_data(expr_te, surv_te, genes)

                    rows = evaluate_three_models(
                        X_tr, surv_tr,
                        X_te, surv_te,
                        fname
                    )

                    for r in rows:
                        all_rows.append({
                            "Cancer": cancer,
                            "FeatureSet": fname,
                            "BaseModel": r.get("ModelType"),
                            "Model": r.get("Model"),
                            "C_index": float(r.get("C_index", np.nan)),
                            "logrank_p": float(r.get("logrank_p", np.nan)),
                            "HR_high_vs_low": float(r.get("HR_high_vs_low", np.nan)),
                            "FeatureNum": int(r.get("FeatureNum", 0)),
                            "Source": "independent_train"
                        })

                except Exception:
                    traceback.print_exc()

    if not all_rows:
        print("⚠️ 无有效结果")
        return None

    df = pd.DataFrame(all_rows)
    safe_csv(df, os.path.join(out_dir, "qubo_vs_fullmodel_results.csv"))

    # 统一 BaseModel 命名
    df["BaseModel"] = df["BaseModel"].replace({
        "RSF-BPCox-Fusion": "Fusion",
        "Fusion": "Fusion",
        "RSF": "RSF",
        "BP-Cox": "BP-Cox"
    })

    # 对比可视化
    plot_qubo_vs_fullmodel(df, out_dir)

    # delta 汇总
    delta_rows = []

    for cancer in sorted(df["Cancer"].unique()):
        for base_model in sorted(df["BaseModel"].dropna().unique()):
            q = df[(df["Cancer"] == cancer) & (df["FeatureSet"] == "QUBO_Full") & (df["BaseModel"] == base_model)]
            f = df[(df["Cancer"] == cancer) & (df["FeatureSet"] == "FullModel") & (df["BaseModel"] == base_model)]

            if q.empty or f.empty:
                continue

            qc = float(q["C_index"].mean())
            fc = float(f["C_index"].mean())

            delta_rows.append({
                "Cancer": cancer,
                "BaseModel": base_model,
                "QUBO_C_index": qc,
                "FullModel_C_index": fc,
                "Delta_QUBO_minus_Full": qc - fc,
                "QUBO_FeatureNum": int(q["FeatureNum"].mean()),
                "FullModel_FeatureNum": int(f["FeatureNum"].mean()),
            })

    if delta_rows:
        delta_df = pd.DataFrame(delta_rows)
        safe_csv(delta_df, os.path.join(out_dir, "qubo_vs_fullmodel_delta.csv"))
        plot_delta(delta_df, out_dir)

    print(f"\n✅ 问题 3 完成：{df.shape[0]} 条结果")
    return df


def plot_qubo_vs_fullmodel(df, out_dir):
    """每个 BaseModel 下，QUBO vs FullModel 的分癌种条形对比"""
    for base_model in sorted(df["BaseModel"].dropna().unique()):
        sub = df[df["BaseModel"] == base_model]

        q = sub[sub["FeatureSet"] == "QUBO_Full"].groupby("Cancer")["C_index"].mean()
        f = sub[sub["FeatureSet"] == "FullModel"].groupby("Cancer")["C_index"].mean()

        common = sorted(set(q.index) & set(f.index))

        if not common:
            continue

        x = np.arange(len(common))
        width = 0.38

        q_vals = [q[c] for c in common]
        f_vals = [f[c] for c in common]

        fig, ax = plt.subplots(figsize=(11, 6))

        ax.bar(x - width / 2, q_vals, width, label="QUBO preselected", color=COLOR_QUBO, alpha=0.85)
        ax.bar(x + width / 2, f_vals, width, label="FullModel (no selection)", color=COLOR_FULL, alpha=0.85)

        for i, (qv, fv) in enumerate(zip(q_vals, f_vals)):
            ax.text(i - width / 2, qv + 0.005, f"{qv:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + width / 2, fv + 0.005, f"{fv:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(common, fontsize=11)
        ax.set_ylabel("Test C-index", fontsize=12, fontweight="bold")
        ax.set_title(f"QUBO preselected vs FullModel: {base_model}", fontsize=14, fontweight="bold")
        ax.set_ylim(0.5, min(0.85, max(max(q_vals), max(f_vals)) + 0.08))
        ax.axhline(0.6, color="gray", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"qubo_vs_fullmodel_{base_model}.png"))
        plt.close()


def plot_delta(delta_df, out_dir):
    """Delta C-index 图：QUBO 减 FullModel"""
    fusion = delta_df[delta_df["BaseModel"] == "Fusion"]

    if fusion.empty:
        fusion = delta_df

    fusion = fusion.sort_values("Delta_QUBO_minus_Full", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6))

    colors = [COLOR_QUBO if v >= 0 else COLOR_FULL for v in fusion["Delta_QUBO_minus_Full"]]

    bars = ax.bar(fusion["Cancer"], fusion["Delta_QUBO_minus_Full"], color=colors, alpha=0.85)

    for bar, v in zip(bars, fusion["Delta_QUBO_minus_Full"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + (0.003 if v >= 0 else -0.003),
            f"{v:+.3f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Delta C-index (QUBO - FullModel)", fontsize=12, fontweight="bold")
    ax.set_title("QUBO preselected minus FullModel (Fusion)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "delta_qubo_minus_fullmodel.png"))
    plt.close()


# =========================================================
# 7. 总报告
# =========================================================

def generate_report(df1, df2, df3):
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results_root": RESULTS_ROOT,
        "output_root": OUTPUT_ROOT,
        "cancers": CANCERS,
        "problem_1_cross_cancer_transfer": {
            "done": df1 is not None,
            "rows": int(df1.shape[0]) if df1 is not None else 0,
        },
        "problem_2_within_cancer_cross_dataset": {
            "done": df2 is not None,
            "rows": int(df2.shape[0]) if df2 is not None else 0,
        },
        "problem_3_qubo_vs_fullmodel": {
            "done": df3 is not None,
            "rows": int(df3.shape[0]) if df3 is not None else 0,
        },
    }

    save_json(report, os.path.join(OUTPUT_ROOT, "analysis_report.json"))

    print("\n报告已保存: analysis_report.json")


# =========================================================
# 8. 主入口
# =========================================================

def main():
    print("=" * 100)
    print("三个问题完整分析 增强版")
    print("=" * 100)
    print("CANCER2_SCRIPT:", CANCER2_SCRIPT)
    print("RESULTS_ROOT:", RESULTS_ROOT)
    print("OUTPUT_ROOT:", OUTPUT_ROOT)
    print("CANCERS:", CANCERS)
    print("FULL_MODEL_GENES:", FULL_MODEL_GENES)

    df1 = None
    df2 = None
    df3 = None

    try:
        df1 = cross_cancer_transfer()
    except Exception:
        traceback.print_exc()

    try:
        df2 = within_cancer_cross_dataset()
    except Exception:
        traceback.print_exc()

    try:
        df3 = qubo_vs_fullmodel()
    except Exception:
        traceback.print_exc()

    try:
        generate_report(df1, df2, df3)
    except Exception:
        traceback.print_exc()

    print("\n" + "=" * 100)
    print("全部完成")
    print("输出目录:", OUTPUT_ROOT)
    print("=" * 100)


if __name__ == "__main__":
    main()
