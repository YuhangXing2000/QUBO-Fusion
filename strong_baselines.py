# -*- coding: utf-8 -*-
"""
癌症_强基线补充.py
只依赖干净的 core_qubo_fusion.py。
新增 RSF_Importance 和 mRMR 两个强基线。
读取已有 Cox 评分 → 选基因 → 训练 RSF/BP/Fusion → 合并旧结果出图。
"""

import os
import argparse
import params
import warnings
import traceback
import importlib.util

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    SNS = True
except Exception:
    SNS = False

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#=========================================================
# 0. 配置（只需改这里）
# =========================================================

CANCER2_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core_qubo_fusion.py")
RESULTS_ROOT = params.OUTPUT_MAIN
OUTPUT_DIR = params.OUTPUT_BASELINES


def parse_args(argv=None):
    """Parse CLI arguments; falls back to defaults above when none are given."""
    parser = argparse.ArgumentParser(description="QUBO-Fusion strong baselines: mRMR + RSF_Importance")
    parser.add_argument("--cancer2_script", default=CANCER2_SCRIPT,
                        help="Path to core_qubo_fusion.py (provides shared model functions)")
    parser.add_argument("--results_root", default=RESULTS_ROOT,
                        help="Output root produced by core_qubo_fusion.py")
    parser.add_argument("--output", dest="output_dir", default=OUTPUT_DIR,
                        help="Output dir for baseline results")
    return parser.parse_args(argv)


def apply_config(args):
    global CANCER2_SCRIPT, RESULTS_ROOT, OUTPUT_DIR
    CANCER2_SCRIPT = args.cancer2_script
    RESULTS_ROOT = args.results_root
    OUTPUT_DIR = args.output_dir


_args = parse_args()
apply_config(_args)


CANCERS     = params.CANCERS
RANDOM_SEED = params.RANDOM_SEED
N_GENES     = params.SELECTED_GENE_NUM
MAX_CAND= params.MAX_QUBO_CANDIDATE_GENES

COLORS = {
    "QUBO_Full":"#E74C3C",
    "QUBO_NoRedundancy":  "#C0392B",
    "FullModel":"#3498DB",
    "RSF_Importance":     "#8E44AD",
    "mRMR":               "#16A085",
    "RandomGenes":        "#95A5A6",
    "TopCoxGenes":        "#F39C12",
}

# =========================================================
# 1. 导入 癌症2.py
# =========================================================

def import_module(path, name="cancer2"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到脚本: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("加载 癌症2.py 函数...")
d2 = import_module(CANCER2_SCRIPT)
d2.RANDOM_SEED = RANDOM_SEED
d2.OUTPUT_ROOT = RESULTS_ROOT

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":130,
    "savefig.dpi":        300,
    "axes.unicode_minus": False,
})


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


_setup_cjk_font()

# =========================================================
# 2. 强基线基因选择
# =========================================================

def select_rsf_importance(expr_train, surv_train, score_df, n_genes=N_GENES):
    """RSF 特征重要性基线，与QUBO 使用相同的Top-MAX_CAND 候选集。"""
    cand = score_df.head(MAX_CAND)["gene"].tolist()
    cand = [g for g in cand if g in expr_train.columns]
    if len(cand) <= n_genes:
        print(f"  RSF_Importance:候选 {len(cand)} <= {n_genes}，返回全部")
        return cand

    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(expr_train[cand].values)
    y = -np.log1p(surv_train["time"].values.astype(float))
    w = np.where(surv_train["event"].values == 1, 1.5, 0.8)

    rf = RandomForestRegressor(
        n_estimators=80, max_depth=8, min_samples_leaf=5,
        random_state=RANDOM_SEED, n_jobs=1
    )
    rf.fit(X, y, sample_weight=w)
    idx = np.argsort(rf.feature_importances_)[::-1][:n_genes]
    selected = [cand[i] for i in idx]
    print(f"  RSF_Importance: {len(selected)} 个基因")
    return selected


def select_mrmr(expr_train, score_df, n_genes=N_GENES):
    """mRMR贪心算法基线，与 QUBO 使用相同的 Top-MAX_CAND 候选集。"""
    cand = score_df.head(MAX_CAND)["gene"].tolist()
    cand = [g for g in cand if g in expr_train.columns]
    if len(cand) <= n_genes:
        print(f"  mRMR: 候选 {len(cand)} <= {n_genes}，返回全部")
        return cand

    score_col = next(
        (c for c in ["score", "z_abs", "cindex"] if c in score_df.columns),
        None
    )
    if score_col:
        smap = dict(zip(score_df["gene"], score_df[score_col]))
        raw = np.array([smap.get(g, 0.0) for g in cand], dtype=float)
    else:
        raw = np.arange(len(cand),0, -1, dtype=float)

    vmin, vmax = raw.min(), raw.max()
    rel = {
        g: float((raw[i] - vmin) / (vmax - vmin + 1e-12))
        for i, g in enumerate(cand)
    }

    X = expr_train[cand].fillna(expr_train[cand].mean())
    corr = X.corr(method="pearson").abs()

    selected = []
    remaining = list(cand)

    for _ in range(n_genes):
        if not remaining:
            break
        if not selected:
            best = max(remaining, key=lambda g: rel[g])
        else:
            def mrmr_score(g):
                red = float(np.mean([
                    corr.loc[g, s] if s in corr.columns else 0.0
                    for s in selected
                ]))
                return rel[g] - red
            best = max(remaining, key=mrmr_score)
        selected.append(best)
        remaining.remove(best)

    print(f"  mRMR: {len(selected)} 个基因")
    return selected


# =========================================================
# 3. 训练 RSF + BP-Cox + Fusion
#    使用 癌症2.py 真实函数名：
#    cv_tune_rsf(X, surv)
#    fit_rsf_predict(X_train, surv_train, X_test, params)
#    cv_tune_bpcox(X, surv)
#    fit_bpcox_predict(X_train, surv_train, X_test, params)
#    tune_fusion_weight_cv(X, surv, rsf_params, bp_params)
#    standardize_risk_by_train(train_risk, test_risk)
#    evaluate_survival_model(name, surv, risk)
# =========================================================

def run_models_for_featureset(tag, genes, expr_train, expr_test, surv_train, surv_test):
    """只跑 RSF / BP-Cox / Fusion，复用 d2 的函数保证实现一致。"""
    genes = [
        g for g in genes
        if g in expr_train.columns and g in expr_test.columns
    ]
    if len(genes) < 3:
        print(f"  ⚠️  {tag} 可用基因数不足，跳过")
        return []

    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    X_tr = scl.fit_transform(imp.fit_transform(expr_train[genes].values))
    X_te = scl.transform(imp.transform(expr_test[genes].values))

    results= []
    rsf_tr= rsf_te   = None
    bp_tr     = bp_te    = None
    rsf_params = None
    bp_params  = None

    # RSF
    try:
        print(f"  [{tag}] RSF")
        rsf_params = d2.cv_tune_rsf(X_tr, surv_train)
        _, rsf_tr, rsf_te = d2.fit_rsf_predict(X_tr, surv_train, X_te, rsf_params)
        res = d2.evaluate_survival_model(f"{tag}-RSF", surv_test, rsf_te)
        res["FeatureSet"] = tag
        res["FeatureNum"] = len(genes)
        results.append(res)
    except Exception:
        traceback.print_exc()

    # BP-Cox
    try:
        print(f"  [{tag}] BP-Cox")
        bp_params = d2.cv_tune_bpcox(X_tr, surv_train)
        _, bp_tr, bp_te = d2.fit_bpcox_predict(X_tr, surv_train, X_te, bp_params)
        res = d2.evaluate_survival_model(f"{tag}-BP-Cox", surv_test, bp_te)
        res["FeatureSet"] = tag
        res["FeatureNum"] = len(genes)
        results.append(res)
    except Exception:
        traceback.print_exc()

    # Fusion
    if rsf_tr is not None and bp_tr is not None:
        try:
            print(f"  [{tag}] Fusion")
            w, w_score, _ = d2.tune_fusion_weight_cv(
                X_tr, surv_train, rsf_params, bp_params
            )
            _, rsf_tr2, rsf_te2 = d2.fit_rsf_predict(
                X_tr, surv_train, X_te, rsf_params
            )
            _, bp_tr2, bp_te2 = d2.fit_bpcox_predict(
                X_tr, surv_train, X_te, bp_params
            )
            _, rsf_te_s = d2.standardize_risk_by_train(rsf_tr2, rsf_te2)
            _, bp_te_s= d2.standardize_risk_by_train(bp_tr2,  bp_te2)
            fusion = w * rsf_te_s + (1 - w) * bp_te_s
            res = d2.evaluate_survival_model(
                f"{tag}-RSF-BPCox-Fusion", surv_test, fusion
            )
            res["FeatureSet"]= tag
            res["FeatureNum"]         = len(genes)
            res["FusionWeight_RSF"]   = w
            res["FusionWeight_BPCox"] = 1 - w
            results.append(res)
        except Exception:
            traceback.print_exc()

    return results


# =========================================================
# 4. 单癌种主流程
# =========================================================

def run_one_cancer(cancer):
    print(f"\n{'='*80}\n癌种: {cancer}\n{'='*80}")

    cancer_out = os.path.join(OUTPUT_DIR, cancer)
    os.makedirs(cancer_out, exist_ok=True)

    # 读取已有 model_results
    old_csv = os.path.join(RESULTS_ROOT, cancer, f"{cancer}_model_results.csv")
    if not os.path.exists(old_csv):
        print(f"⚠️  找不到: {old_csv}，跳过")
        return None
    old_df = pd.read_csv(old_csv)
    print(f"已有结果: {len(old_df)} 行")

    # 读取已有 univariate_cox_scores
    score_csv = os.path.join(
        RESULTS_ROOT, cancer, f"{cancer}_train_univariate_cox_scores.csv"
    )
    if not os.path.exists(score_csv):
        print(f"⚠️  找不到 Cox 评分: {score_csv}，跳过")
        return None
    score_df = pd.read_csv(score_csv)

    # 加载数据
    # load_cancer_data 返回元组: (merged_expr, surv_df, cohort_labels, gene_list)
    try:
        loaded = d2.load_cancer_data(cancer)
    except Exception:
        traceback.print_exc()
        print(f"⚠️  {cancer} 数据加载失败，跳过")
        return None

    if loaded is None:
        print(f"⚠️  {cancer} 返回空数据，跳过")
        return None

    expr= loaded[0]
    surv_df = loaded[1]

    # 与原实验相同的划分（按索引）
    try:
        strat = (
            surv_df["event"].values
            if pd.Series(surv_df["event"].values).value_counts().min() >= 2
            else None
        )
    except Exception:
        strat = None

    n = len(expr)
    idx = np.arange(n)
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.3, random_state=RANDOM_SEED, stratify=strat
    )

    expr_train  = expr.iloc[idx_tr].reset_index(drop=True)
    expr_test   = expr.iloc[idx_te].reset_index(drop=True)
    surv_train  = surv_df.iloc[idx_tr].reset_index(drop=True)
    surv_test   = surv_df.iloc[idx_te].reset_index(drop=True)
    print(f"训练集 {expr_train.shape}, 测试集 {expr_test.shape}")

    # 两个强基线的基因集
    feature_sets = {}
    try:
        feature_sets["RSF_Importance"] = select_rsf_importance(
            expr_train, surv_train, score_df
        )
    except Exception:
        traceback.print_exc()
    try:
        feature_sets["mRMR"] = select_mrmr(expr_train, score_df)
    except Exception:
        traceback.print_exc()

    if not feature_sets:
        print("⚠️  基线基因选择全部失败")
        return old_df

    # 保存新基因集
    for tag, genes in feature_sets.items():
        pd.DataFrame({"gene": genes}).to_csv(
            os.path.join(cancer_out, f"{cancer}_{tag}_genes.csv"),
            index=False, encoding="utf-8-sig"
        )

    # 训练
    new_rows = []
    for tag, genes in feature_sets.items():
        rows = run_models_for_featureset(
            tag, genes, expr_train, expr_test, surv_train, surv_test
        )
        new_rows.extend(rows)
        print(f"  {tag}: 完成 {len(rows)} 个模型")

    if not new_rows:
        print("⚠️  新基线训练全部失败")
        return old_df

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([old_df, new_df], ignore_index=True)

    ci_col = next(
        (c for c in ["C_index", "Cindex", "c_index"] if c in combined.columns),
        None
    )
    if ci_col:
        combined = combined.sort_values(ci_col, ascending=False).reset_index(drop=True)

    combined.to_csv(
        os.path.join(cancer_out, f"{cancer}_model_results_with_baselines.csv"),
        index=False, encoding="utf-8-sig"
    )
    print(f"✅  {cancer} 完成，共 {len(combined)} 行")
    return combined



# =========================================================
# 5. 汇总可视化
# =========================================================

def plot_baseline_comparison(all_results):
    fusion_rows = []
    for cancer, df in all_results.items():
        if df is None or df.empty:
            continue
        # 找 Model 列和 C_index 列
        mcol = next(
            (c for c in ["Model", "model", "ModelName"] if c in df.columns), None
        )
        ccol = next(
            (c for c in ["C_index", "Cindex", "c_index"] if c in df.columns), None
        )
        if mcol is None or ccol is None:
            continue
        sub = df[df[mcol].astype(str).str.contains("Fusion", na=False)]
        for _, row in sub.iterrows():
            fusion_rows.append({
                "Cancer":cancer,
                "FeatureSet": row.get("FeatureSet", ""),
                "C_index":    row.get(ccol, np.nan),
            })

    if not fusion_rows:
        print("⚠️  无 Fusion 结果，跳过汇总图")
        return

    fdf = pd.DataFrame(fusion_rows)

    feature_order = [
        "QUBO_NoRedundancy", "QUBO_Full",
        "RSF_Importance", "mRMR",
        "TopCoxGenes", "RandomGenes", "FullModel",
    ]
    feature_sets = [f for f in feature_order if f in fdf["FeatureSet"].values]
    cancers = sorted(fdf["Cancer"].unique())
    n_fs = len(feature_sets)
    n_ca = len(cancers)
    x = np.arange(n_ca)
    width = 0.8/ max(n_fs, 1)

    # 图1：分组柱状图
    fig, ax = plt.subplots(figsize=(max(14, n_ca * 2.2), 7))
    for i, fs in enumerate(feature_sets):
        vals = [
            fdf[(fdf["Cancer"] == c) & (fdf["FeatureSet"] == fs)]["C_index"].mean()
            for c in cancers
        ]
        offset = (i - n_fs / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width * 0.9,
            label=fs, color=COLORS.get(fs, "#BDC3C7"), alpha=0.85
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold"
                )

    ax.set_xticks(x)
    ax.set_xticklabels(cancers, fontsize=12)
    ax.set_ylabel("C-index（Fusion 模型）", fontsize=12)
    ax.set_title("六癌种：QUBO vs强基线 vs弱基线（Fusion）",
                 fontsize=14, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="随机基线(0.5)")
    ax.legend(fontsize=9, ncol=4, loc="upper right")
    ax.set_ylim(0.45, min(0.85, fdf["C_index"].max() + 0.08))
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "fig1_comparison_all_baselines.png"),
        bbox_inches="tight"
    )
    plt.close()
    print("✅  图1 已保存")

    # 图2：Delta（QUBO_NoRed 相对各方法）
    ref_fs = "QUBO_NoRedundancy"
    compare_fss = [
        f for f in ["RSF_Importance", "mRMR", "FullModel", "TopCoxGenes"]
        if f in fdf["FeatureSet"].values
    ]

    if ref_fs in fdf["FeatureSet"].values and compare_fss:
        fig, axes = plt.subplots(
            1, len(compare_fss),
            figsize=(5 * len(compare_fss), 6),
            sharey=True
        )
        if len(compare_fss) == 1:
            axes = [axes]

        for ax, fs in zip(axes, compare_fss):
            deltas, labels = [], []
            for c in cancers:
                rv = fdf[(fdf["Cancer"] == c) & (fdf["FeatureSet"] == ref_fs)]["C_index"].mean()
                cv = fdf[(fdf["Cancer"] == c) & (fdf["FeatureSet"] == fs)]["C_index"].mean()
                if not (np.isnan(rv) or np.isnan(cv)):
                    deltas.append(rv - cv)
                    labels.append(c)

            bar_colors = [
                COLORS["QUBO_NoRedundancy"] if d >= 0 else COLORS.get(fs, "#3498DB")
                for d in deltas
            ]
            ax.bar(labels, deltas, color=bar_colors, alpha=0.85)
            for i, v in enumerate(deltas):
                ax.text(
                    i, v + (0.003 if v >= 0 else -0.003),
                    f"{v:+.3f}",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=9, fontweight="bold"
                )
            ax.axhline(0, color="black", lw=1)
            ax.set_title(f"QUBO_NoRed − {fs}", fontsize=11, fontweight="bold")
            ax.set_ylabel("Delta C-index" if fs == compare_fss[0] else "")
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(
            "QUBO_NoRedundancy 相对其他方法的优势（Fusion，正值=QUBO更好）",
            fontsize=13, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, "fig2_delta_qubo_vs_baselines.png"),
            bbox_inches="tight"
        )
        plt.close()
        print("✅  图2 已保存")

    # 图3：热图
    pivot = fdf.pivot_table(
        index="FeatureSet", columns="Cancer",
        values="C_index", aggfunc="mean"
    )
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(
        figsize=(max(10, n_ca * 1.5), max(5, len(pivot) * 0.8))
    )
    if SNS:
        sns.heatmap(
            pivot, annot=True, fmt=".3f",
            cmap="RdYlGn", vmin=0.50, vmax=0.72,
            linewidths=0.5, linecolor="white",
            ax=ax, cbar_kws={"label": "C-index"}
        )
    else:
        im = ax.imshow(
            pivot.values, cmap="RdYlGn",
            vmin=0.50, vmax=0.72, aspect="auto"
        )
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(
                        j, i, f"{v:.3f}",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="black" if v < 0.64 else "white"
                    )

    ax.set_title("各特征选择方案 × 癌种 C-index 热图（Fusion）",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("癌种", fontsize=11)
    ax.set_ylabel("特征选择方案", fontsize=11)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "fig3_heatmap_all_featuresets.png"),
        bbox_inches="tight"
    )
    plt.close()
    print("✅  图3 已保存")

    fdf.to_csv(
        os.path.join(OUTPUT_DIR, "all_cancers_fusion_summary.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("✅  汇总 CSV 已保存")


# =========================================================
# 6. 主入口
# =========================================================

def main():
    print("=" * 80)
    print("强基线补充实验：RSF_Importance + mRMR")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 80)

    all_results = {}
    for cancer in CANCERS:
        try:
            all_results[cancer] = run_one_cancer(cancer)
        except Exception:
            traceback.print_exc()
            print(f"❌  {cancer} 失败，继续")
            all_results[cancer] = None

    print("\n生成汇总图...")
    plot_baseline_comparison(all_results)
    print(f"\n全部完成，输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
