# -*- coding: utf-8 -*-
"""
QUBO-Fusion 在 BRCA 中的专项强基线验证
====================================

功能：
1. QUBO_Full 与 QUBO_NoRed 特征选择
2. TopCox、Random、mRMR、RF_Importance、Lasso_Selected、ElasticNet_Selected、VarTop500 强基线
3. CoxPH、Lasso-Cox、PCA-Cox、RSF、BP-Cox、RSF-BP-Cox Fusion
4. 多随机种子 7:3 同分布验证
5. C-index、HR、HR 95% CI、log-rank p
6. Fusion 权重通过训练集内部交叉验证搜索
7. LassoCox 预测模型通过训练集内部 CV 选择 alpha
8. QUBO 基因集合 Jaccard 稳定性
9. 特征冗余度分析
10. QUBO_NoRed-Fusion 相对所有组合配对差值
11. 所有 FeatureSet × Model 组合排名
12. seed=42 测试集 KM 曲线，避免全样本训练全样本评估泄漏
13. 中文图表和 Markdown 总结报告

依赖：
pip install numpy pandas scipy scikit-learn lifelines matplotlib seaborn torch openpyxl
pip install scikit-survival
可选：
pip install pyreadr

运行：
python brca_validation.py --data_root data --output results/brca
"""

import os
import sys
import json
import math
import random
import argparse
import params
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ttest_rel, wilcoxon

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# ============================================================
# 1. 配置
# ============================================================

DEFAULT_SEEDS = params.BRCA_SEEDS

FEATURE_SET_ORDER = [
    "QUBO_Full",
    "QUBO_NoRed",
    "TopCox",
    "Random",
    "mRMR",
    "RF_Importance",
    "Lasso_Selected",
    "ElasticNet_Selected",
    "VarTop500",
]

MODEL_ORDER = [
    "CoxPH",
    "LassoCox",
    "PCA-Cox",
    "RSF",
    "BP-Cox",
    "Fusion",
]

COLORS = {
    "QUBO_Full": "#D73027",
    "QUBO_NoRed": "#A50026",
    "TopCox": "#1A9850",
    "Random": "#969696",
    "mRMR": "#FDAE61",
    "RF_Importance": "#756BB1",
    "Lasso_Selected": "#66C2A5",
    "ElasticNet_Selected": "#3288BD",
    "VarTop500": "#3182BD",

    "CoxPH": "#1F78B4",
    "LassoCox": "#33A02C",
    "PCA-Cox": "#6A3D9A",
    "RSF": "#E31A1C",
    "BP-Cox": "#FF7F00",
    "Fusion": "#B15928",
}


# ============================================================
# 2. 随机性和字体
# ============================================================

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_chinese_font():
    import matplotlib.font_manager as fm

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
        "PingFang SC",
    ]

    available = {f.name for f in fm.fontManager.ttflist}
    selected = next((x for x in candidates if x in available), None)

    if selected is None:
        print("[警告] 未检测到常见中文字体，图中中文可能无法正常显示。")
        selected = "DejaVu Sans"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [selected, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
    })

    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", context="notebook")
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    print(f"[字体] 使用: {selected}")


# ============================================================
# 3. 数据读取
# ============================================================

def read_table(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, index_col=0)

    if suffix in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t", index_col=0)

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, index_col=0)

    raise ValueError(f"不支持的数据格式: {path}")


def locate_files(brca_dir):
    expression_files = []
    clinical_files = []

    for p in Path(brca_dir).iterdir():
        if not p.is_file():
            continue

        name = p.name.lower()
        suffix = p.suffix.lower()

        if suffix not in [".csv", ".tsv", ".txt", ".xlsx", ".xls", ".rda", ".rds"]:
            continue

        if any(x in name for x in ["expr", "expression", "mrna", "rna", "tpm", "fpkm"]):
            expression_files.append(p)

        if any(x in name for x in ["clin", "surv", "clinical", "pheno", "metadata"]):
            clinical_files.append(p)

    return expression_files, clinical_files


def _normalize_col_name(x):
    return str(x).strip().lower().replace("-", "_").replace(".", "_").replace(" ", "_")


def _normalize_event_value(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip().lower().replace('"', "").replace("'", "")
    s = s.replace(" ", "_").replace("-", "_")

    positive = [
        "1", "true", "dead", "deceased", "death", "event", "yes", "y",
        "progressed", "progression", "recurred", "recurrence", "relapse",
        "relapsed", "died", "failure", "failed", "with_event"
    ]

    negative = [
        "0", "false", "alive", "censored", "censor", "no", "n",
        "non_event", "not_progressed", "no_recurrence", "disease_free",
        "non_relapse", "without_event", "living"
    ]

    if s in positive:
        return 1

    if s in negative:
        return 0

    if any(k in s for k in ["dead", "deceased", "death", "relapse", "progress", "recur"]):
        return 1

    if any(k in s for k in ["alive", "censor", "living", "disease_free"]):
        return 0

    try:
        v = float(s)
        return 1 if v > 0 else 0
    except Exception:
        return np.nan


def _pick_survival_columns(df):
    cols = list(df.columns)
    norm = {c: _normalize_col_name(c) for c in cols}
    endpoints = ["os", "pfs", "dfs", "rfs", "dss"]

    for ep in endpoints:
        tcs = []
        ecs = []

        for c in cols:
            nc = norm[c]

            if nc in [f"{ep}_time", f"{ep}time", f"{ep}_days", f"{ep}_months"]:
                tcs.append(c)

            if nc in [ep, f"{ep}_status", f"{ep}_event"]:
                ecs.append(c)

        if tcs and ecs:
            return tcs[0], ecs[0]

    time_kws = [
        "time", "days", "day", "month", "months", "year", "years",
        "survival", "follow", "followup", "follow_up"
    ]

    event_kws = [
        "status", "event", "censor", "death", "dead", "vital",
        "recurrence", "progression", "relapse"
    ]

    tcs = [
        c for c in cols
        if any(k in norm[c] for k in time_kws)
        and not any(k in norm[c] for k in ["sample", "patient", "barcode", "id"])
    ]

    ecs = [
        c for c in cols
        if any(k in norm[c] for k in event_kws)
        and not any(k in norm[c] for k in ["sample", "patient", "barcode", "id"])
    ]

    for tc in tcs:
        for ec in ecs:
            if tc != ec:
                return tc, ec

    return None, None


def _find_sample_id_col(df):
    cols = list(df.columns)
    norm = {c: _normalize_col_name(c) for c in cols}

    exact = [
        "sample", "sample_id", "barcode", "patient", "patient_id",
        "id", "geo_accession", "gsm", "case_id", "submitter_id"
    ]

    for c in cols:
        if norm[c] in exact:
            return c

    for c in cols:
        if any(k in norm[c] for k in ["sample", "patient", "barcode", "geo_accession", "gsm"]):
            return c

    return None


def _prepare_clinical_from_raw(clin):
    clin = clin.copy()

    sid_col = _find_sample_id_col(clin)

    if sid_col is not None:
        clin.index = clin[sid_col].astype(str)

    time_col, event_col = _pick_survival_columns(clin)

    if time_col is None or event_col is None:
        print("[错误] 无法自动识别临床 time/event 列")
        print("[临床列名]", list(clin.columns))
        return None

    out = pd.DataFrame(index=clin.index.astype(str))
    out["time"] = pd.to_numeric(clin[time_col], errors="coerce")
    out["event"] = clin[event_col].map(_normalize_event_value)

    # 如果时间看起来像月份，也可以不转换；这里保持原单位，只要求正数
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["time", "event"])
    out = out[out["time"] > 0]

    out["event"] = out["event"].astype(int)

    return out


def _standardize_expression_numeric(expr):
    expr = expr.copy()

    if expr.shape[1] > 0:
        first_col = expr.columns[0]
        first_name = str(first_col).lower()

        if any(x in first_name for x in ["gene", "symbol", "probe", "id"]):
            expr = expr.set_index(first_col)

    expr.index = expr.index.astype(str)
    expr.columns = expr.columns.astype(str)
    expr = expr.loc[:, ~expr.columns.duplicated()]
    expr = expr.replace([np.inf, -np.inf], np.nan)
    expr = expr.apply(pd.to_numeric, errors="coerce")

    return expr


def _match_expr_clin(expr, clin):
    """
    尝试 expr 原方向和转置方向，选择与临床样本 ID 交集最多的方向。
    返回样本×基因表达矩阵和临床表。
    """

    candidates = []

    for candidate_expr in [expr, expr.T]:
        e = candidate_expr.copy()
        e.index = e.index.astype(str)
        e.columns = e.columns.astype(str)

        expr_ids = pd.Index(e.index.astype(str))
        clin_ids = pd.Index(clin.index.astype(str))

        common = expr_ids.intersection(clin_ids)
        method = "exact"

        if len(common) < 80:
            expr_short = pd.Series(expr_ids.astype(str)).str[:12]
            clin_short = pd.Series(clin_ids.astype(str)).str[:12]

            expr_map = {}
            for original, short in zip(expr_ids, expr_short):
                expr_map.setdefault(short, original)

            clin_map = {}
            for original, short in zip(clin_ids, clin_short):
                clin_map.setdefault(short, original)

            common_short = sorted(set(expr_map.keys()) & set(clin_map.keys()))

            if len(common_short) >= len(common):
                e2 = e.copy()
                c2 = clin.copy()

                e2["_sid"] = e2.index.astype(str).str[:12]
                c2["_sid"] = c2.index.astype(str).str[:12]

                e2 = e2.drop_duplicates("_sid").set_index("_sid")
                c2 = c2.drop_duplicates("_sid").set_index("_sid")

                common2 = pd.Index(common_short)

                candidates.append((len(common2), e2, c2, common2, "first12"))
                continue

        candidates.append((len(common), e, clin.copy(), common, method))

    candidates = sorted(candidates, key=lambda z: z[0], reverse=True)

    best_n, best_expr, best_clin, common, method = candidates[0]

    if best_n < 80:
        raise RuntimeError(f"表达矩阵和临床表有效交集只有 {best_n} 个样本")

    best_expr = best_expr.loc[common].copy()
    best_clin = best_clin.loc[common].copy()

    valid_cols = best_expr.notna().sum(axis=0) >= max(5, int(0.8 * len(best_expr)))
    best_expr = best_expr.loc[:, valid_cols]
    best_expr = best_expr.loc[:, best_expr.nunique(dropna=True) > 1]

    if best_expr.shape[1] < 100:
        raise RuntimeError(f"有效基因数只有 {best_expr.shape[1]}")

    print(f"[数据] ID匹配方式: {method}")
    print(f"[数据] 对齐后样本数: {len(best_expr)}")
    print(f"[数据] 有效基因数: {best_expr.shape[1]}")
    print(f"[数据] 事件数: {int(best_clin['event'].sum())}")
    print(f"[数据] 事件率: {best_clin['event'].mean():.2%}")

    return best_expr, best_clin


def load_from_rda_or_rds(path):
    try:
        import pyreadr
    except Exception as exc:
        raise RuntimeError(
            "读取 RDA/RDS 需要 pyreadr，请执行：pip install pyreadr，"
            "或提供 csv/tsv/xlsx 格式表达和临床文件。"
        ) from exc

    result = pyreadr.read_r(str(path))
    dfs = []

    for name, obj in result.items():
        if isinstance(obj, pd.DataFrame):
            dfs.append((name, obj))

    if len(dfs) < 2:
        raise RuntimeError(f"RDA/RDS 中未找到至少两个 DataFrame: {path}")

    expr_candidates = []
    clin_candidates = []

    for name, df in dfs:
        lname = name.lower()
        cols_lower = [str(c).lower() for c in df.columns]

        if any(k in lname for k in ["expr", "expression", "mrna", "rna"]):
            expr_candidates.append((name, df))

        if any(k in lname for k in ["clin", "surv", "clinical", "pheno"]):
            clin_candidates.append((name, df))

        if any(k in " ".join(cols_lower) for k in ["status", "event", "survival", "time"]):
            clin_candidates.append((name, df))

    if not expr_candidates:
        expr_candidates = sorted(dfs, key=lambda z: z[1].shape[0] * z[1].shape[1], reverse=True)[:1]

    if not clin_candidates:
        clin_candidates = sorted(dfs, key=lambda z: z[1].shape[0] * z[1].shape[1])[:1]

    expr_name, expr = expr_candidates[0]
    clin_name, clin = clin_candidates[0]

    print(f"[数据] R文件表达对象: {expr_name}, shape={expr.shape}")
    print(f"[数据] R文件临床对象: {clin_name}, shape={clin.shape}")

    return expr, clin


def load_brca_data(data_root):
    brca_dir = Path(data_root) / "BRCA"

    if not brca_dir.exists():
        raise FileNotFoundError(f"BRCA目录不存在: {brca_dir}")

    expression_files, clinical_files = locate_files(brca_dir)

    tab_expr = [p for p in expression_files if p.suffix.lower() not in [".rda", ".rds"]]
    tab_clin = [p for p in clinical_files if p.suffix.lower() not in [".rda", ".rds"]]

    if tab_expr and tab_clin:
        expr = read_table(tab_expr[0])
        clin = read_table(tab_clin[0])
        print(f"[数据] 表达文件: {tab_expr[0]}")
        print(f"[数据] 临床文件: {tab_clin[0]}")

    else:
        r_files = list(brca_dir.glob("*.rda")) + list(brca_dir.glob("*.rds"))

        if not r_files:
            raise FileNotFoundError(f"{brca_dir}中未找到可识别的数据文件。")

        r_prioritized = [
            p for p in r_files
            if any(k in p.name.lower() for k in ["mrna", "expr", "expression"])
        ]

        r_file = r_prioritized[0] if r_prioritized else r_files[0]
        expr, clin = load_from_rda_or_rds(r_file)
        print(f"[数据] R文件: {r_file}")

    expr = _standardize_expression_numeric(expr)
    clin = _prepare_clinical_from_raw(clin)

    if clin is None:
        raise RuntimeError("临床表解析失败")

    expr, clin = _match_expr_clin(expr, clin)

    return expr, clin


# ============================================================
# 4. 生存评估和单变量评分
# ============================================================

def safe_cindex(time, event, risk):
    risk = np.asarray(risk, dtype=float)

    if len(np.unique(event)) < 2:
        return np.nan

    if not np.isfinite(risk).all():
        return np.nan

    try:
        return float(concordance_index(time, -risk, event))
    except Exception:
        return np.nan


def univariate_cox_scores(expr, time, event):
    """
    只在训练集内计算单变量 Cox 综合评分：
    composite = 0.50 * |z| rank + 0.30 * single-gene C-index rank + 0.20 * variance rank
    """

    rows = []

    for gene in expr.columns:
        x = pd.to_numeric(expr[gene], errors="coerce")

        if x.nunique(dropna=True) < 2:
            continue

        df = pd.DataFrame({
            "x": x.values,
            "time": time,
            "event": event,
        }).replace([np.inf, -np.inf], np.nan).dropna()

        if len(df) < 30 or df["event"].sum() < 5:
            continue

        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(df, duration_col="time", event_col="event")

            z = float(abs(cph.summary.loc["x", "z"]))
            beta = float(cph.params_["x"])

            single_c = safe_cindex(
                df["time"].values,
                df["event"].values,
                beta * df["x"].values,
            )

            variance = float(df["x"].var())

            rows.append({
                "gene": gene,
                "z_abs": z,
                "single_cindex": single_c,
                "variance": variance,
                "beta": beta,
            })

        except Exception:
            continue

    if not rows:
        raise RuntimeError("所有基因的单变量Cox均失败。")

    result = pd.DataFrame(rows).set_index("gene")

    for col in ["z_abs", "single_cindex", "variance"]:
        result[col + "_rank"] = result[col].rank(pct=True)

    result["composite"] = (
        0.50 * result["z_abs_rank"]
        + 0.30 * result["single_cindex_rank"]
        + 0.20 * result["variance_rank"]
    )

    return result.sort_values("composite", ascending=False)


# ============================================================
# 5. 特征选择
# ============================================================

def simulated_annealing(
    Q,
    target_k,
    seed=42,
    n_iter=4000,
    t_start=5.0,
    t_end=0.01,
):
    """
    固定基数 QUBO 模拟退火：
    通过 1-0 交换保持选中特征数恒定，不显式加入基数惩罚项。
    """

    rng = np.random.default_rng(seed)
    n = Q.shape[0]

    if target_k >= n:
        return np.ones(n, dtype=int)

    x = np.zeros(n, dtype=int)
    x[rng.choice(n, target_k, replace=False)] = 1

    def energy(vector):
        return float(vector @ Q @ vector)

    current_energy = energy(x)
    best_x = x.copy()
    best_energy = current_energy

    alpha = (t_end / t_start) ** (1.0 / n_iter)
    temperature = t_start

    for _ in range(n_iter):
        ones = np.flatnonzero(x == 1)
        zeros = np.flatnonzero(x == 0)

        if len(ones) == 0 or len(zeros) == 0:
            break

        i = rng.choice(ones)
        j = rng.choice(zeros)

        x_new = x.copy()
        x_new[i] = 0
        x_new[j] = 1

        new_energy = energy(x_new)
        delta = new_energy - current_energy

        if delta <= 0:
            accept = True
        else:
            accept = rng.random() < np.exp(-delta / max(temperature, 1e-12))

        if accept:
            x = x_new
            current_energy = new_energy

            if current_energy < best_energy:
                best_x = x.copy()
                best_energy = current_energy

        temperature *= alpha

    return best_x


def qubo_select(
    expr,
    time,
    event,
    n_select=20,
    n_candidates=80,
    w_rel=1.0,
    w_red=0.35,
    seed=42,
):
    scores = univariate_cox_scores(expr, time, event)

    candidates = scores.head(
        min(n_candidates, len(scores))
    ).index.tolist()

    if len(candidates) <= n_select:
        return candidates, scores

    x = expr[candidates].copy()
    x = x.fillna(x.median())

    corr = x.corr().fillna(0).values
    n = len(candidates)

    Q = np.zeros((n, n), dtype=float)

    for i, gene in enumerate(candidates):
        Q[i, i] = -w_rel * float(scores.loc[gene, "composite"])

    for i in range(n):
        for j in range(i + 1, n):
            redundancy = float(corr[i, j] ** 2)
            Q[i, j] += w_red * redundancy
            Q[j, i] += w_red * redundancy

    selected = simulated_annealing(
        Q=Q,
        target_k=n_select,
        seed=seed,
    )

    genes = [
        candidates[i]
        for i in range(n)
        if selected[i] == 1
    ]

    if len(genes) != n_select:
        genes = candidates[:n_select]

    return genes, scores


def top_cox_select(expr, time, event, n_select=20):
    scores = univariate_cox_scores(expr, time, event)
    return scores.head(n_select).index.tolist()


def random_select(expr, n_select=20, seed=42):
    rng = np.random.default_rng(seed)
    n_select = min(n_select, expr.shape[1])
    return list(rng.choice(expr.columns, n_select, replace=False))


def mrmr_select(expr, time, event, n_select=20):
    scores = univariate_cox_scores(expr, time, event)
    candidates = scores.head(min(80, len(scores))).index.tolist()

    if len(candidates) <= n_select:
        return candidates

    x = expr[candidates].fillna(expr[candidates].median())
    corr = x.corr().abs().fillna(0)

    selected = []
    remaining = candidates.copy()

    while remaining and len(selected) < n_select:
        if not selected:
            gene = remaining[0]
        else:
            values = []

            for gene in remaining:
                relevance = float(scores.loc[gene, "composite"])
                redundancy = float(corr.loc[gene, selected].mean())
                values.append((relevance - 0.35 * redundancy, gene))

            gene = max(values, key=lambda z: z[0])[1]

        selected.append(gene)
        remaining.remove(gene)

    return selected


def rf_importance_select(expr, time, event, n_select=20, seed=42):
    scores = univariate_cox_scores(expr, time, event)
    candidates = scores.head(min(80, len(scores))).index.tolist()

    if len(candidates) <= n_select:
        return candidates

    x = expr[candidates].fillna(expr[candidates].median()).values

    target = -np.log1p(time)
    sample_weight = np.where(event == 1, 1.5, 0.8)

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=1,
    )

    rf.fit(x, target, sample_weight=sample_weight)

    importance = pd.Series(
        rf.feature_importances_,
        index=candidates,
    ).sort_values(ascending=False)

    return importance.head(n_select).index.tolist()


def coxnet_feature_select_cv(
    expr,
    time,
    event,
    n_select=20,
    l1_ratio=1.0,
    seed=42,
    n_candidates=300,
    n_splits=3,
):
    """
    Coxnet 强特征选择基线。

    l1_ratio=1.0 : Lasso-Cox 特征选择
    l1_ratio=0.5 : ElasticNet-Cox 特征选择
    """

    try:
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.util import Surv
    except ImportError:
        print("[警告] scikit-survival 不存在，Coxnet特征选择退化为TopCox")
        return top_cox_select(expr, time, event, n_select=n_select)

    scores = univariate_cox_scores(expr, time, event)
    candidates = scores.head(min(n_candidates, len(scores))).index.tolist()

    if len(candidates) <= n_select:
        return candidates

    x_df = expr[candidates].copy()
    x_df = x_df.replace([np.inf, -np.inf], np.nan)
    x_df = x_df.fillna(x_df.median())

    x = x_df.values.astype(float)
    y_event = event.astype(bool)
    y_time = time.astype(float)

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    alpha_score_map = {}

    for fold_id, (tr_idx, va_idx) in enumerate(splitter.split(x, event)):
        x_tr_raw = x[tr_idx]
        x_va_raw = x[va_idx]

        t_tr = y_time[tr_idx]
        e_tr = y_event[tr_idx]
        t_va = y_time[va_idx]
        e_va = y_event[va_idx]

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        x_tr = imputer.fit_transform(x_tr_raw)
        x_va = imputer.transform(x_va_raw)

        x_tr = scaler.fit_transform(x_tr)
        x_va = scaler.transform(x_va)

        y_tr = Surv.from_arrays(e_tr, t_tr)

        try:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                alpha_min_ratio=0.01,
                n_alphas=80,
                max_iter=10000,
            )

            model.fit(x_tr, y_tr)

            coefs = model.coef_
            alphas = model.alphas_

            if coefs.ndim == 1:
                coefs = coefs.reshape(-1, 1)

            for j, alpha in enumerate(alphas):
                coef = coefs[:, j]

                if np.all(np.abs(coef) < 1e-12):
                    continue

                risk = x_va @ coef
                cidx = safe_cindex(t_va, e_va.astype(int), risk)

                if np.isfinite(cidx):
                    alpha = float(alpha)
                    alpha_score_map.setdefault(alpha, []).append(cidx)

        except Exception as exc:
            print(f"[警告] Coxnet特征选择CV fold={fold_id}失败: {exc}")
            continue

    if not alpha_score_map:
        print("[警告] Coxnet特征选择CV失败，退化为TopCox")
        return top_cox_select(expr, time, event, n_select=n_select)

    mean_alpha_scores = {
        alpha: float(np.nanmean(values))
        for alpha, values in alpha_score_map.items()
        if len(values) > 0
    }

    best_alpha = max(mean_alpha_scores, key=mean_alpha_scores.get)

    y_full = Surv.from_arrays(y_event, y_time)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x_full = imputer.fit_transform(x)
    x_full = scaler.fit_transform(x_full)

    try:
        final_model = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio,
            alphas=[best_alpha],
            max_iter=10000,
        )

        final_model.fit(x_full, y_full)

        coef = final_model.coef_

        if coef.ndim == 2:
            coef = coef[:, 0]

        coef_abs = pd.Series(
            np.abs(coef),
            index=candidates,
        ).sort_values(ascending=False)

        nonzero = coef_abs[coef_abs > 1e-10]

        if len(nonzero) >= n_select:
            selected = nonzero.head(n_select).index.tolist()

        elif len(nonzero) > 0:
            selected = nonzero.index.tolist()
            remain = [
                g for g in scores.index.tolist()
                if g not in selected
            ]
            selected += remain[: max(0, n_select - len(selected))]

        else:
            selected = scores.head(n_select).index.tolist()

        return selected[:n_select]

    except Exception as exc:
        print(f"[警告] Coxnet完整训练失败，退化为TopCox: {exc}")
        return top_cox_select(expr, time, event, n_select=n_select)


def lasso_selected_genes(expr, time, event, n_select=20, seed=42):
    return coxnet_feature_select_cv(
        expr=expr,
        time=time,
        event=event,
        n_select=n_select,
        l1_ratio=1.0,
        seed=seed,
        n_candidates=300,
        n_splits=3,
    )


def elasticnet_selected_genes(expr, time, event, n_select=20, seed=42):
    return coxnet_feature_select_cv(
        expr=expr,
        time=time,
        event=event,
        n_select=n_select,
        l1_ratio=0.5,
        seed=seed,
        n_candidates=300,
        n_splits=3,
    )


def vartop500_select(expr, n_select=500):
    """
    大特征基线：训练集内方差 Top500。
    注意：这不是严格意义的全基因模型，而是 VarTop500 大特征基线。
    """
    variance = expr.var().sort_values(ascending=False)
    return variance.head(min(n_select, len(variance))).index.tolist()


# ============================================================
# 6. 预测模型
# ============================================================

def scale_train_test(x_train, x_test):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train = imputer.fit_transform(x_train)
    test = imputer.transform(x_test)

    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    return train, test, imputer, scaler


def fit_coxph(x_train, time_train, event_train, x_test):
    model = CoxPHFitter(penalizer=0.01)

    train = pd.DataFrame(x_train)
    train["time"] = time_train
    train["event"] = event_train

    model.fit(train, duration_col="time", event_col="event")

    test = pd.DataFrame(x_test)
    risk = model.predict_partial_hazard(test).values.ravel()

    return risk


def fit_lasso_cox(
    x_train,
    time_train,
    event_train,
    x_test,
    seed=42,
    n_splits=3,
):
    """
    Lasso-Cox 预测模型。
    使用训练集内部 CV 选择 alpha，避免直接取 Coxnet 路径最后一个 alpha。
    """

    try:
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.util import Surv
    except ImportError as exc:
        raise RuntimeError(
            "Lasso-Cox需要scikit-survival，请执行：pip install scikit-survival"
        ) from exc

    y_event = event_train.astype(bool)
    y_time = time_train.astype(float)

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    alpha_score_map = {}

    for fold_id, (tr_idx, va_idx) in enumerate(splitter.split(x_train, event_train)):
        x_tr = x_train[tr_idx]
        x_va = x_train[va_idx]

        t_tr = y_time[tr_idx]
        e_tr = y_event[tr_idx]
        t_va = y_time[va_idx]
        e_va = y_event[va_idx]

        y_tr = Surv.from_arrays(e_tr, t_tr)

        try:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=1.0,
                alpha_min_ratio=0.01,
                n_alphas=80,
                max_iter=10000,
            )

            model.fit(x_tr, y_tr)

            coefs = model.coef_
            alphas = model.alphas_

            if coefs.ndim == 1:
                coefs = coefs.reshape(-1, 1)

            for j, alpha in enumerate(alphas):
                coef = coefs[:, j]

                if np.all(np.abs(coef) < 1e-12):
                    continue

                risk = x_va @ coef
                cidx = safe_cindex(t_va, e_va.astype(int), risk)

                if np.isfinite(cidx):
                    alpha = float(alpha)
                    alpha_score_map.setdefault(alpha, []).append(cidx)

        except Exception as exc:
            print(f"[警告] LassoCox CV fold={fold_id}失败: {exc}")
            continue

    y_full = Surv.from_arrays(y_event, y_time)

    if not alpha_score_map:
        model = CoxnetSurvivalAnalysis(
            l1_ratio=1.0,
            alpha_min_ratio=0.01,
            n_alphas=80,
            max_iter=10000,
        )

        model.fit(x_train, y_full)
        coef = model.coef_

        if coef.ndim == 2:
            coef = coef[:, -1]

        return x_test @ coef

    mean_alpha_scores = {
        alpha: float(np.nanmean(values))
        for alpha, values in alpha_score_map.items()
        if len(values) > 0
    }

    best_alpha = max(mean_alpha_scores, key=mean_alpha_scores.get)

    final_model = CoxnetSurvivalAnalysis(
        l1_ratio=1.0,
        alphas=[best_alpha],
        max_iter=10000,
    )

    final_model.fit(x_train, y_full)

    coef = final_model.coef_

    if coef.ndim == 2:
        coef = coef[:, 0]

    risk = x_test @ coef

    return risk


def fit_pca_cox(x_train, time_train, event_train, x_test):
    n_components = min(
        8,
        x_train.shape[1],
        max(1, x_train.shape[0] - 1),
    )

    pca = PCA(n_components=n_components, random_state=42)

    train_pca = pca.fit_transform(x_train)
    test_pca = pca.transform(x_test)

    return fit_coxph(
        train_pca,
        time_train,
        event_train,
        test_pca,
    )


def fit_rsf_model(x_train, time_train, event_train, x_test, seed):
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
    except ImportError as exc:
        raise RuntimeError(
            "RSF需要scikit-survival，请执行：pip install scikit-survival"
        ) from exc

    y = Surv.from_arrays(
        event_train.astype(bool),
        time_train.astype(float),
    )

    model = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=1,
        random_state=seed,
    )

    model.fit(x_train, y)

    return model.predict(x_test)


def cox_partial_likelihood_loss(risk, time, event):
    order = torch.argsort(time, descending=True)

    risk = risk[order]
    event = event[order]

    log_cumsum = torch.logcumsumexp(risk, dim=0)
    losses = -event * (risk - log_cumsum)

    return losses.sum() / torch.clamp(event.sum(), min=1.0)


def fit_bpcox_model(
    x_train,
    time_train,
    event_train,
    x_test,
    seed=42,
    epochs=150,
):
    if not HAS_TORCH:
        raise RuntimeError("BP-Cox需要torch，请执行：pip install torch")

    import torch
    import torch.nn as nn

    set_seed(seed)

    class BPCox(nn.Module):
        def __init__(self, input_dim):
            super().__init__()

            hidden = min(64, max(16, input_dim * 2))

            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(hidden, max(8, hidden // 2)),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(max(8, hidden // 2), 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = BPCox(x_train.shape[1])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    x = torch.tensor(x_train, dtype=torch.float32)
    t = torch.tensor(time_train, dtype=torch.float32)
    e = torch.tensor(event_train, dtype=torch.float32)

    best_state = None
    best_loss = float("inf")
    patience = 25
    no_improve = 0

    model.train()

    for _ in range(epochs):
        optimizer.zero_grad()

        risk = model(x)
        loss = cox_partial_likelihood_loss(risk, t, e)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        value = float(loss.detach().cpu())

        if value < best_loss:
            best_loss = value
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            no_improve = 0

        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():
        test_tensor = torch.tensor(x_test, dtype=torch.float32)
        risk = model(test_tensor).cpu().numpy()

    return risk


def fit_base_risk(model_name, x_train, time_train, event_train, x_test, seed):
    if model_name == "CoxPH":
        return fit_coxph(
            x_train,
            time_train,
            event_train,
            x_test,
        )

    if model_name == "LassoCox":
        return fit_lasso_cox(
            x_train,
            time_train,
            event_train,
            x_test,
            seed=seed,
        )

    if model_name == "PCA-Cox":
        return fit_pca_cox(
            x_train,
            time_train,
            event_train,
            x_test,
        )

    if model_name == "RSF":
        return fit_rsf_model(
            x_train,
            time_train,
            event_train,
            x_test,
            seed,
        )

    if model_name == "BP-Cox":
        return fit_bpcox_model(
            x_train,
            time_train,
            event_train,
            x_test,
            seed,
        )

    raise ValueError(f"未知模型: {model_name}")


# ============================================================
# 7. Fusion
# ============================================================

def zscore_by_train(train_risk, test_risk):
    train_risk = np.asarray(train_risk, dtype=float)
    test_risk = np.asarray(test_risk, dtype=float)

    mean = float(np.mean(train_risk))
    std = float(np.std(train_risk))

    if std < 1e-8:
        std = 1.0

    return (
        (train_risk - mean) / std,
        (test_risk - mean) / std,
    )


def train_two_risks(
    x_train,
    time_train,
    event_train,
    x_valid,
    seed,
):
    rsf_train = fit_base_risk(
        "RSF",
        x_train,
        time_train,
        event_train,
        x_train,
        seed,
    )

    rsf_valid = fit_base_risk(
        "RSF",
        x_train,
        time_train,
        event_train,
        x_valid,
        seed,
    )

    bp_train = fit_base_risk(
        "BP-Cox",
        x_train,
        time_train,
        event_train,
        x_train,
        seed,
    )

    bp_valid = fit_base_risk(
        "BP-Cox",
        x_train,
        time_train,
        event_train,
        x_valid,
        seed,
    )

    rsf_train, rsf_valid = zscore_by_train(rsf_train, rsf_valid)
    bp_train, bp_valid = zscore_by_train(bp_train, bp_valid)

    return rsf_train, rsf_valid, bp_train, bp_valid


def select_fusion_weight_cv(
    x_train,
    time_train,
    event_train,
    seed,
    folds=3,
):
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )

    weights = np.arange(0.0, 1.01, 0.05)
    scores = {float(w): [] for w in weights}

    for fold_id, (tr_idx, va_idx) in enumerate(splitter.split(x_train, event_train)):
        x_tr = x_train[tr_idx]
        x_va = x_train[va_idx]

        t_tr = time_train[tr_idx]
        e_tr = event_train[tr_idx]

        t_va = time_train[va_idx]
        e_va = event_train[va_idx]

        try:
            _, rsf_va, _, bp_va = train_two_risks(
                x_tr,
                t_tr,
                e_tr,
                x_va,
                seed + fold_id + 100,
            )

            for weight in weights:
                fused = weight * rsf_va + (1.0 - weight) * bp_va
                value = safe_cindex(t_va, e_va, fused)

                if np.isfinite(value):
                    scores[float(weight)].append(value)

        except Exception as exc:
            print(f"[警告] Fusion CV fold={fold_id}失败: {exc}")
            continue

    mean_scores = {
        weight: np.nanmean(values) if values else np.nan
        for weight, values in scores.items()
    }

    valid = {
        weight: value
        for weight, value in mean_scores.items()
        if np.isfinite(value)
    }

    if not valid:
        return 0.5, mean_scores

    best_weight = max(valid, key=valid.get)

    return float(best_weight), mean_scores


def fit_fusion(
    x_train,
    time_train,
    event_train,
    x_test,
    seed,
):
    best_weight, cv_scores = select_fusion_weight_cv(
        x_train,
        time_train,
        event_train,
        seed,
        folds=3,
    )

    rsf_train = fit_base_risk(
        "RSF",
        x_train,
        time_train,
        event_train,
        x_train,
        seed,
    )

    rsf_test = fit_base_risk(
        "RSF",
        x_train,
        time_train,
        event_train,
        x_test,
        seed,
    )

    bp_train = fit_base_risk(
        "BP-Cox",
        x_train,
        time_train,
        event_train,
        x_train,
        seed,
    )

    bp_test = fit_base_risk(
        "BP-Cox",
        x_train,
        time_train,
        event_train,
        x_test,
        seed,
    )

    rsf_train, rsf_test = zscore_by_train(rsf_train, rsf_test)
    bp_train, bp_test = zscore_by_train(bp_train, bp_test)

    fused_test = (
        best_weight * rsf_test
        + (1.0 - best_weight) * bp_test
    )

    return fused_test, best_weight, cv_scores


# ============================================================
# 8. 生存统计
# ============================================================

def survival_statistics(time, event, risk):
    risk = np.asarray(risk, dtype=float)

    cutoff = float(np.nanmedian(risk))
    group = (risk >= cutoff).astype(int)

    high = group == 1
    low = group == 0

    result = {
        "HR": np.nan,
        "HR_95CI_Low": np.nan,
        "HR_95CI_High": np.nan,
        "LogRankP": np.nan,
        "HighRiskN": int(high.sum()),
        "LowRiskN": int(low.sum()),
        "RiskCutoff": cutoff,
    }

    if high.sum() < 3 or low.sum() < 3:
        return result

    try:
        lr = logrank_test(
            time[high],
            time[low],
            event_observed_A=event[high],
            event_observed_B=event[low],
        )

        result["LogRankP"] = float(lr.p_value)

    except Exception:
        pass

    try:
        df = pd.DataFrame({
            "time": time,
            "event": event.astype(int),
            "risk_group": group,
        })

        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(
            df,
            duration_col="time",
            event_col="event",
        )

        coef = float(cph.params_["risk_group"])
        ci = cph.confidence_intervals_.loc["risk_group"].values

        result["HR"] = float(np.exp(coef))
        result["HR_95CI_Low"] = float(np.exp(ci[0]))
        result["HR_95CI_High"] = float(np.exp(ci[1]))

    except Exception:
        pass

    return result


# ============================================================
# 9. 单次实验
# ============================================================

def build_feature_sets(x_train, time_train, event_train, seed):
    qubo_full, full_scores = qubo_select(
        x_train,
        time_train,
        event_train,
        n_select=20,
        n_candidates=80,
        w_rel=1.0,
        w_red=0.35,
        seed=seed,
    )

    qubo_nored, nored_scores = qubo_select(
        x_train,
        time_train,
        event_train,
        n_select=20,
        n_candidates=80,
        w_rel=1.0,
        w_red=0.0,
        seed=seed,
    )

    topcox = top_cox_select(
        x_train,
        time_train,
        event_train,
        n_select=20,
    )

    random_genes = random_select(
        x_train,
        n_select=20,
        seed=seed,
    )

    mrmr = mrmr_select(
        x_train,
        time_train,
        event_train,
        n_select=20,
    )

    rf_imp = rf_importance_select(
        x_train,
        time_train,
        event_train,
        n_select=20,
        seed=seed,
    )

    lasso_sel = lasso_selected_genes(
        x_train,
        time_train,
        event_train,
        n_select=20,
        seed=seed,
    )

    elasticnet_sel = elasticnet_selected_genes(
        x_train,
        time_train,
        event_train,
        n_select=20,
        seed=seed,
    )

    vartop500 = vartop500_select(
        x_train,
        n_select=min(500, x_train.shape[1]),
    )

    feature_sets = {
        "QUBO_Full": qubo_full,
        "QUBO_NoRed": qubo_nored,
        "TopCox": topcox,
        "Random": random_genes,
        "mRMR": mrmr,
        "RF_Importance": rf_imp,
        "Lasso_Selected": lasso_sel,
        "ElasticNet_Selected": elasticnet_sel,
        "VarTop500": vartop500,
    }

    return feature_sets, full_scores, nored_scores


def run_one_seed(expr, clin, seed):
    set_seed(seed)

    time = clin["time"].values.astype(float)
    event = clin["event"].values.astype(int)

    idx_train, idx_test = train_test_split(
        np.arange(len(expr)),
        test_size=0.30,
        random_state=seed,
        stratify=event,
    )

    x_train_df = expr.iloc[idx_train]
    x_test_df = expr.iloc[idx_test]

    time_train = time[idx_train]
    event_train = event[idx_train]

    time_test = time[idx_test]
    event_test = event[idx_test]

    feature_sets, full_scores, nored_scores = build_feature_sets(
        x_train_df,
        time_train,
        event_train,
        seed,
    )

    results = []
    selected_rows = []

    for feature_name in FEATURE_SET_ORDER:
        genes = [
            g for g in feature_sets[feature_name]
            if g in x_train_df.columns
        ]

        selected_rows.append({
            "Seed": seed,
            "FeatureSet": feature_name,
            "Genes": "|".join(genes),
            "N_genes": len(genes),
        })

        if len(genes) < 2:
            continue

        xtr, xte, _, _ = scale_train_test(
            x_train_df[genes].values,
            x_test_df[genes].values,
        )

        for model_name in MODEL_ORDER:
            fusion_weight = np.nan

            try:
                if model_name == "Fusion":
                    risk, fusion_weight, _ = fit_fusion(
                        xtr,
                        time_train,
                        event_train,
                        xte,
                        seed,
                    )

                else:
                    risk = fit_base_risk(
                        model_name,
                        xtr,
                        time_train,
                        event_train,
                        xte,
                        seed,
                    )

                cindex = safe_cindex(
                    time_test,
                    event_test,
                    risk,
                )

                stats = survival_statistics(
                    time_test,
                    event_test,
                    risk,
                )

                results.append({
                    "Seed": seed,
                    "FeatureSet": feature_name,
                    "Model": model_name,
                    "C_index": cindex,
                    "HR": stats["HR"],
                    "HR_95CI_Low": stats["HR_95CI_Low"],
                    "HR_95CI_High": stats["HR_95CI_High"],
                    "LogRankP": stats["LogRankP"],
                    "HighRiskN": stats["HighRiskN"],
                    "LowRiskN": stats["LowRiskN"],
                    "RiskCutoff": stats["RiskCutoff"],
                    "N_genes": len(genes),
                    "TrainN": len(idx_train),
                    "TestN": len(idx_test),
                    "TrainEvents": int(event_train.sum()),
                    "TestEvents": int(event_test.sum()),
                    "FusionWeight": fusion_weight,
                })

            except Exception as exc:
                print(
                    f"[警告] seed={seed}, "
                    f"{feature_name}+{model_name}失败: {exc}"
                )

                results.append({
                    "Seed": seed,
                    "FeatureSet": feature_name,
                    "Model": model_name,
                    "C_index": np.nan,
                    "HR": np.nan,
                    "HR_95CI_Low": np.nan,
                    "HR_95CI_High": np.nan,
                    "LogRankP": np.nan,
                    "HighRiskN": np.nan,
                    "LowRiskN": np.nan,
                    "RiskCutoff": np.nan,
                    "N_genes": len(genes),
                    "TrainN": len(idx_train),
                    "TestN": len(idx_test),
                    "TrainEvents": int(event_train.sum()),
                    "TestEvents": int(event_test.sum()),
                    "FusionWeight": fusion_weight,
                })

    result_df = pd.DataFrame(results)
    selected_df = pd.DataFrame(selected_rows)

    return result_df, selected_df, feature_sets


# ============================================================
# 10. 汇总统计
# ============================================================

def ci95(series):
    values = pd.to_numeric(series, errors="coerce").dropna().values

    if len(values) < 2:
        return np.nan, np.nan

    mean = np.mean(values)
    half = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))

    return mean - half, mean + half


def summarize_results(df):
    rows = []

    for (feature, model), sub in df.groupby(["FeatureSet", "Model"]):
        vals = sub["C_index"].dropna()

        low, high = ci95(vals)

        rows.append({
            "FeatureSet": feature,
            "Model": model,
            "N": len(vals),
            "C_index_Mean": vals.mean() if len(vals) else np.nan,
            "C_index_SD": vals.std(ddof=1) if len(vals) > 1 else np.nan,
            "C_index_Median": vals.median() if len(vals) else np.nan,
            "C_index_CI95_Low": low,
            "C_index_CI95_High": high,
            "HR_Median": sub["HR"].median(),
            "LogRankP_Median": sub["LogRankP"].median(),
            "FusionWeight_Mean": sub["FusionWeight"].mean(),
        })

    return pd.DataFrame(rows)


def paired_comparison(df, target="QUBO_NoRed", model="Fusion"):
    pivot = df[df["Model"] == model].pivot_table(
        index="Seed",
        columns="FeatureSet",
        values="C_index",
        aggfunc="mean",
    )

    if target not in pivot.columns:
        return pd.DataFrame()

    rows = []

    for baseline in pivot.columns:
        if baseline == target:
            continue

        sub = pivot[[target, baseline]].dropna()

        if len(sub) < 3:
            continue

        diff = sub[target] - sub[baseline]

        mean_diff = float(diff.mean())
        sd_diff = float(diff.std(ddof=1))
        median_diff = float(diff.median())
        win_rate = float((diff > 0).mean())

        half = 1.96 * sd_diff / np.sqrt(len(diff))
        ci_low = mean_diff - half
        ci_high = mean_diff + half

        try:
            t_p = float(ttest_rel(sub[target], sub[baseline]).pvalue)
        except Exception:
            t_p = np.nan

        try:
            w_p = float(wilcoxon(diff).pvalue)
        except Exception:
            w_p = np.nan

        rows.append({
            "Target": f"{target}-{model}",
            "Baseline": f"{baseline}-{model}",
            "N": len(diff),
            "MeanDifference": mean_diff,
            "MedianDifference": median_diff,
            "SDDifference": sd_diff,
            "CI95_Low": ci_low,
            "CI95_High": ci_high,
            "WinRate": win_rate,
            "Paired_t_p": t_p,
            "Wilcoxon_p": w_p,
        })

    return pd.DataFrame(rows).sort_values(
        "MeanDifference",
        ascending=False,
    ).reset_index(drop=True)


def ranking_table(summary):
    if summary.empty:
        return pd.DataFrame()

    df = summary.copy()
    df = df.sort_values("C_index_Mean", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)

    cols = [
        "Rank",
        "FeatureSet",
        "Model",
        "N",
        "C_index_Mean",
        "C_index_SD",
        "C_index_CI95_Low",
        "C_index_CI95_High",
        "HR_Median",
        "LogRankP_Median",
        "FusionWeight_Mean",
    ]

    cols = [c for c in cols if c in df.columns]

    return df[cols]


def target_combo_advantage(
    df,
    target_feature="QUBO_NoRed",
    target_model="Fusion",
):
    target = df[
        (df["FeatureSet"] == target_feature)
        & (df["Model"] == target_model)
    ][["Seed", "C_index"]].rename(columns={"C_index": "Target_C_index"})

    if target.empty:
        return pd.DataFrame()

    rows = []

    for (feature, model), sub in df.groupby(["FeatureSet", "Model"]):
        if feature == target_feature and model == target_model:
            continue

        base = sub[["Seed", "C_index"]].rename(
            columns={"C_index": "Baseline_C_index"}
        )

        merged = target.merge(base, on="Seed", how="inner")
        merged = merged.dropna()

        if len(merged) < 3:
            continue

        diff = merged["Target_C_index"] - merged["Baseline_C_index"]

        mean_diff = float(diff.mean())
        sd_diff = float(diff.std(ddof=1))
        median_diff = float(diff.median())
        win_rate = float((diff > 0).mean())

        ci_half = 1.96 * sd_diff / np.sqrt(len(diff))
        ci_low = mean_diff - ci_half
        ci_high = mean_diff + ci_half

        try:
            t_p = float(ttest_rel(
                merged["Target_C_index"],
                merged["Baseline_C_index"],
                nan_policy="omit",
            ).pvalue)
        except Exception:
            t_p = np.nan

        try:
            w_p = float(wilcoxon(diff).pvalue)
        except Exception:
            w_p = np.nan

        rows.append({
            "Target": f"{target_feature}-{target_model}",
            "Baseline": f"{feature}-{model}",
            "N": len(diff),
            "MeanDifference": mean_diff,
            "MedianDifference": median_diff,
            "SDDifference": sd_diff,
            "CI95_Low": ci_low,
            "CI95_High": ci_high,
            "WinRate": win_rate,
            "Paired_t_p": t_p,
            "Wilcoxon_p": w_p,
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "MeanDifference",
            ascending=False,
        ).reset_index(drop=True)

    return result


def gene_jaccard(selected_df):
    rows = []

    for feature in ["QUBO_Full", "QUBO_NoRed"]:
        sub = selected_df[selected_df["FeatureSet"] == feature].copy()

        gene_sets = []

        for _, row in sub.iterrows():
            genes = set(str(row["Genes"]).split("|"))
            genes = {g for g in genes if g and g != "nan"}
            gene_sets.append((row["Seed"], genes))

        for i in range(len(gene_sets)):
            for j in range(i + 1, len(gene_sets)):
                seed_i, set_i = gene_sets[i]
                seed_j, set_j = gene_sets[j]

                union = set_i | set_j
                inter = set_i & set_j

                jac = len(inter) / len(union) if union else np.nan

                rows.append({
                    "FeatureSet": feature,
                    "SeedA": seed_i,
                    "SeedB": seed_j,
                    "Jaccard": jac,
                    "IntersectionN": len(inter),
                    "UnionN": len(union),
                })

    return pd.DataFrame(rows)


def redundancy_summary(selected_df, expr):
    rows = []

    for _, row in selected_df.iterrows():
        genes = [
            g for g in str(row["Genes"]).split("|")
            if g in expr.columns
        ]

        if len(genes) < 2:
            continue

        x = expr[genes].copy()
        x = x.fillna(x.median())

        corr = x.corr().abs().fillna(0).values
        upper = corr[np.triu_indices_from(corr, k=1)]

        if len(upper) == 0:
            continue

        rows.append({
            "Seed": row["Seed"],
            "FeatureSet": row["FeatureSet"],
            "N_genes": len(genes),
            "MeanAbsCorrelation": np.nanmean(upper),
            "MedianAbsCorrelation": np.nanmedian(upper),
            "MaxAbsCorrelation": np.nanmax(upper),
        })

    return pd.DataFrame(rows)


# ============================================================
# 11. 图表
# ============================================================

def savefig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(summary, fig_dir):
    table = summary.pivot_table(
        index="FeatureSet",
        columns="Model",
        values="C_index_Mean",
    )

    table = table.reindex(
        index=[x for x in FEATURE_SET_ORDER if x in table.index],
        columns=[x for x in MODEL_ORDER if x in table.columns],
    )

    fig, ax = plt.subplots(figsize=(13, 8))

    if HAS_SEABORN:
        sns.heatmap(
            table,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0.45,
            vmax=0.75,
            linewidths=0.5,
            ax=ax,
        )

    else:
        im = ax.imshow(table.values, cmap="RdYlGn", vmin=0.45, vmax=0.75)
        fig.colorbar(im, ax=ax)

    ax.set_title("BRCA各特征集与预测模型的平均C-index")
    ax.set_xlabel("预测模型")
    ax.set_ylabel("特征集")

    savefig(fig, fig_dir / "图1_Cindex热图.png")


def plot_fusion_summary(summary, fig_dir):
    sub = summary[summary["Model"] == "Fusion"].copy()
    sub = sub.set_index("FeatureSet").reindex(FEATURE_SET_ORDER)
    sub = sub.dropna(subset=["C_index_Mean"])

    x = np.arange(len(sub))
    y = sub["C_index_Mean"].values
    low = sub["C_index_CI95_Low"].values
    high = sub["C_index_CI95_High"].values

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = [COLORS.get(x, "#555555") for x in sub.index]

    ax.bar(
        x,
        y,
        color=colors,
        alpha=0.85,
        yerr=[
            y - low,
            high - y,
        ],
        capsize=5,
    )

    for i, value in enumerate(y):
        ax.text(
            i,
            value + 0.005,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
        )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(sub.index, rotation=30, ha="right")
    ax.set_ylabel("C-index")
    ax.set_title("BRCA中Fusion模型的多种子平均性能及95%置信区间")

    savefig(fig, fig_dir / "图2_Fusion多种子比较.png")


def plot_paired_difference(paired, fig_dir):
    if paired.empty:
        return

    paired = paired.sort_values("MeanDifference")

    y = np.arange(len(paired))
    mean = paired["MeanDifference"].values
    low = paired["CI95_Low"].values
    high = paired["CI95_High"].values

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.errorbar(
        mean,
        y,
        xerr=[
            mean - low,
            high - mean,
        ],
        fmt="o",
        color="#D73027",
        ecolor="#555555",
        capsize=4,
    )

    ax.axvline(0, color="black", linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(paired["Baseline"])
    ax.set_xlabel("C-index差值")
    ax.set_title("QUBO_NoRed-Fusion相对Fusion特征基线的配对差值")

    savefig(fig, fig_dir / "图3_Fusion配对差值.png")


def plot_jaccard(jaccard, fig_dir):
    if jaccard.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    groups = []
    labels = []

    for feature in ["QUBO_Full", "QUBO_NoRed"]:
        values = jaccard[
            jaccard["FeatureSet"] == feature
        ]["Jaccard"].dropna().values

        if len(values):
            groups.append(values)
            labels.append(feature)

    if groups:
        ax.boxplot(groups, labels=labels, patch_artist=True)
        ax.set_ylabel("Jaccard相似度")
        ax.set_title("QUBO基因集合跨随机种子稳定性")

    savefig(fig, fig_dir / "图4_QUBO基因Jaccard稳定性.png")


def plot_redundancy(redundancy, fig_dir):
    if redundancy.empty:
        return

    summary = redundancy.groupby("FeatureSet")[
        "MeanAbsCorrelation"
    ].agg(["mean", "std"]).reindex(FEATURE_SET_ORDER).dropna()

    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(len(summary))

    ax.bar(
        x,
        summary["mean"],
        yerr=summary["std"],
        capsize=4,
        color=[
            COLORS.get(name, "#555555")
            for name in summary.index
        ],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=30, ha="right")
    ax.set_ylabel("基因间平均绝对相关系数")
    ax.set_title("不同特征选择方法的基因冗余度")

    savefig(fig, fig_dir / "图5_基因冗余度比较.png")


def plot_stability(summary, fig_dir):
    sub = summary[summary["Model"] == "Fusion"].copy()
    sub = sub.set_index("FeatureSet").reindex(FEATURE_SET_ORDER)
    sub = sub.dropna(subset=["C_index_SD"])

    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(len(sub))

    ax.bar(
        x,
        sub["C_index_SD"],
        color=[
            COLORS.get(name, "#555555")
            for name in sub.index
        ],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(sub.index, rotation=30, ha="right")
    ax.set_ylabel("C-index标准差")
    ax.set_title("Fusion模型跨随机种子的稳定性")

    savefig(fig, fig_dir / "图7_Fusion稳定性.png")


def plot_km_seed42_testset(
    expr,
    clin,
    output_dir,
    feature_sets,
    target_feature="QUBO_NoRed",
):
    seed = 42

    if target_feature not in feature_sets:
        print(f"[警告] feature_sets中没有 {target_feature}，跳过KM")
        return

    genes = [
        g for g in feature_sets[target_feature]
        if g in expr.columns
    ]

    if len(genes) < 2:
        print("[警告] KM基因数不足，跳过")
        return

    time = clin["time"].values.astype(float)
    event = clin["event"].values.astype(int)

    idx_train, idx_test = train_test_split(
        np.arange(len(expr)),
        test_size=0.30,
        random_state=seed,
        stratify=event,
    )

    x_train_df = expr.iloc[idx_train][genes]
    x_test_df = expr.iloc[idx_test][genes]

    time_train = time[idx_train]
    event_train = event[idx_train]

    time_test = time[idx_test]
    event_test = event[idx_test]

    xtr, xte, _, _ = scale_train_test(
        x_train_df.values,
        x_test_df.values,
    )

    try:
        risk_test, fusion_weight, _ = fit_fusion(
            xtr,
            time_train,
            event_train,
            xte,
            seed,
        )

    except Exception as exc:
        print(f"[警告] seed=42测试集KM曲线生成失败: {exc}")
        return

    cindex = safe_cindex(time_test, event_test, risk_test)
    stats = survival_statistics(time_test, event_test, risk_test)

    group = risk_test >= np.median(risk_test)

    fig, ax = plt.subplots(figsize=(8, 6))

    for value, label, color in [
        (False, "低风险组", "#3182BD"),
        (True, "高风险组", "#D73027"),
    ]:
        mask = group == value

        if mask.sum() < 3:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(
            time_test[mask],
            event_observed=event_test[mask],
            label=f"{label} n={mask.sum()}",
        )

        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            color=color,
        )

    title = (
        f"BRCA seed=42 测试集 {target_feature}-Fusion KM曲线\n"
        f"C-index={cindex:.3f}, HR={stats['HR']:.3f}, "
        f"log-rank p={stats['LogRankP']:.2e}, "
        f"Fusion w={fusion_weight:.2f}"
    )

    ax.set_title(title)
    ax.set_xlabel("随访时间")
    ax.set_ylabel("生存概率")
    ax.legend()

    savefig(
        fig,
        output_dir / f"图6_seed42测试集_{target_feature}_Fusion_KM曲线.png",
    )

    km_stats = pd.DataFrame([{
        "Seed": seed,
        "FeatureSet": target_feature,
        "Model": "Fusion",
        "C_index": cindex,
        "HR": stats["HR"],
        "HR_95CI_Low": stats["HR_95CI_Low"],
        "HR_95CI_High": stats["HR_95CI_High"],
        "LogRankP": stats["LogRankP"],
        "HighRiskN": stats["HighRiskN"],
        "LowRiskN": stats["LowRiskN"],
        "FusionWeight": fusion_weight,
        "N_genes": len(genes),
    }])

    km_stats.to_csv(
        output_dir / f"seed42测试集_{target_feature}_Fusion_KM统计.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 12. Markdown 报告
# ============================================================

def _round_df(df, cols, ndigits=4):
    out = df.copy()

    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(ndigits)

    return out


def write_report(
    output_dir,
    data_info,
    summary,
    paired,
    jaccard,
    redundancy,
    ranking=None,
    target_advantage=None,
):
    report_path = output_dir / "BRCA强基线验证报告.md"

    lines = []

    lines.append("# QUBO-Fusion在BRCA中的专项强基线验证\n")

    lines.append("## 一、数据概况\n")
    lines.append(f"- 样本数：{data_info['n_samples']}\n")
    lines.append(f"- 基因数：{data_info['n_genes']}\n")
    lines.append(f"- 事件数：{data_info['n_events']}\n")
    lines.append(f"- 事件率：{data_info['event_rate']:.2%}\n")

    lines.append("\n## 二、平均C-index结果\n")

    if not summary.empty:
        display = _round_df(
            summary,
            [
                "C_index_Mean",
                "C_index_SD",
                "C_index_Median",
                "C_index_CI95_Low",
                "C_index_CI95_High",
                "HR_Median",
                "LogRankP_Median",
                "FusionWeight_Mean",
            ],
        )

        lines.append(display.to_markdown(index=False))
        lines.append("\n")

    lines.append("## 三、所有组合平均C-index排名\n")

    if ranking is not None and not ranking.empty:
        display_rank = _round_df(
            ranking,
            [
                "C_index_Mean",
                "C_index_SD",
                "C_index_CI95_Low",
                "C_index_CI95_High",
                "HR_Median",
                "LogRankP_Median",
                "FusionWeight_Mean",
            ],
        )

        lines.append(display_rank.head(30).to_markdown(index=False))
        lines.append("\n")

    else:
        lines.append("没有可用排名结果。\n")

    lines.append("## 四、QUBO_NoRed-Fusion相对Fusion特征基线的配对差值\n")

    if paired.empty:
        lines.append("没有足够的配对结果。\n")

    else:
        display_paired = _round_df(
            paired,
            [
                "MeanDifference",
                "MedianDifference",
                "SDDifference",
                "CI95_Low",
                "CI95_High",
                "WinRate",
                "Paired_t_p",
                "Wilcoxon_p",
            ],
        )

        lines.append(display_paired.to_markdown(index=False))
        lines.append("\n")

    lines.append("## 五、QUBO_NoRed-Fusion相对所有组合的配对差值\n")

    if target_advantage is not None and not target_advantage.empty:
        display_adv = _round_df(
            target_advantage,
            [
                "MeanDifference",
                "MedianDifference",
                "SDDifference",
                "CI95_Low",
                "CI95_High",
                "WinRate",
                "Paired_t_p",
                "Wilcoxon_p",
            ],
        )

        lines.append(display_adv.to_markdown(index=False))
        lines.append("\n")

    else:
        lines.append("没有足够的目标组合比较结果。\n")

    lines.append("## 六、基因集合稳定性\n")

    if not jaccard.empty:
        j_summary = jaccard.groupby("FeatureSet")["Jaccard"].agg(
            ["mean", "std", "median"]
        )

        j_summary = j_summary.round(4)
        lines.append(j_summary.to_markdown())
        lines.append("\n")

    else:
        lines.append("没有可用Jaccard结果。\n")

    lines.append("## 七、基因冗余度\n")

    if not redundancy.empty:
        r_summary = redundancy.groupby("FeatureSet")[
            "MeanAbsCorrelation"
        ].agg(["mean", "std", "median"])

        r_summary = r_summary.round(4)
        lines.append(r_summary.to_markdown())
        lines.append("\n")

    else:
        lines.append("没有可用冗余度结果。\n")

    lines.append("## 八、解释边界\n")
    lines.append(
        "本验证用于评估BRCA中QUBO-Fusion的性能、稳定性、强基线对比、"
        "特征压缩和冗余控制表现。需要注意：同分布7:3随机划分仍属于方法潜力上界验证，"
        "不能替代独立外部队列验证。VarTop500是方差Top500大特征基线，"
        "不是严格意义上的全基因组模型。QUBO实现采用固定基数交换搜索，"
        "而非显式加入基数惩罚项。\n"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return report_path


# ============================================================
# 13. 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="QUBO-Fusion BRCA专项强基线验证"
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=params.DATA_ROOT,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--n_seeds",
        type=int,
        default=params.BRCA_N_SEEDS,
    )

    parser.add_argument(
        "--skip_km",
        action="store_true",
    )

    args = parser.parse_args()

    setup_chinese_font()

    data_root = Path(args.data_root)

    if args.output is None:
        output_dir = Path(params.OUTPUT_BRCA)
    else:
        output_dir = Path(args.output)

    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"

    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("QUBO-Fusion在BRCA中的专项强基线验证")
    print("=" * 80)

    expr, clin = load_brca_data(data_root)

    data_info = {
        "n_samples": int(len(expr)),
        "n_genes": int(expr.shape[1]),
        "n_events": int(clin["event"].sum()),
        "event_rate": float(clin["event"].mean()),
    }

    seeds = DEFAULT_SEEDS[:max(1, min(args.n_seeds, len(DEFAULT_SEEDS)))]

    all_results = []
    all_selected = []

    feature_sets_seed42 = None

    for seed in seeds:
        print(f"\n[运行] seed={seed}")

        result_df, selected_df, feature_sets = run_one_seed(
            expr,
            clin,
            seed,
        )

        all_results.append(result_df)
        all_selected.append(selected_df)

        if seed == 42:
            feature_sets_seed42 = feature_sets

        print(
            f"[完成] seed={seed}, "
            f"有效结果={result_df['C_index'].notna().sum()}"
        )

    df_results = pd.concat(all_results, ignore_index=True)
    df_selected = pd.concat(all_selected, ignore_index=True)

    df_summary = summarize_results(df_results)

    df_ranking = ranking_table(df_summary)

    df_paired = paired_comparison(
        df_results,
        target="QUBO_NoRed",
        model="Fusion",
    )

    df_target_advantage = target_combo_advantage(
        df_results,
        target_feature="QUBO_NoRed",
        target_model="Fusion",
    )

    df_jaccard = gene_jaccard(df_selected)
    df_redundancy = redundancy_summary(df_selected, expr)

    df_results.to_csv(
        result_dir / "brca_all_seed_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df_selected.to_csv(
        result_dir / "brca_selected_genes_per_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df_summary.to_csv(
        result_dir / "brca_summary_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df_ranking.to_csv(
        result_dir / "brca_all_combo_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df_paired.to_csv(
        result_dir / "brca_fusion_paired_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df_target_advantage.to_csv(
        result_dir / "brca_QUBO_NoRed_Fusion_vs_all_combos.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df_jaccard.to_csv(
        result_dir / "brca_gene_jaccard.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df_redundancy.to_csv(
        result_dir / "brca_redundancy_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_heatmap(df_summary, figure_dir)
    plot_fusion_summary(df_summary, figure_dir)
    plot_paired_difference(df_paired, figure_dir)
    plot_jaccard(df_jaccard, figure_dir)
    plot_redundancy(df_redundancy, figure_dir)
    plot_stability(df_summary, figure_dir)

    if not args.skip_km and feature_sets_seed42 is not None:
        plot_km_seed42_testset(
            expr=expr,
            clin=clin,
            output_dir=figure_dir,
            feature_sets=feature_sets_seed42,
            target_feature="QUBO_NoRed",
        )

    report_path = write_report(
        output_dir,
        data_info,
        df_summary,
        df_paired,
        df_jaccard,
        df_redundancy,
        ranking=df_ranking,
        target_advantage=df_target_advantage,
    )

    print("\n" + "=" * 80)
    print("验证完成")
    print("=" * 80)
    print(f"结果目录: {result_dir}")
    print(f"图表目录: {figure_dir}")
    print(f"报告文件: {report_path}")

    fusion = df_summary[
        df_summary["Model"] == "Fusion"
    ].copy()

    if not fusion.empty:
        print("\n[Fusion平均C-index]")
        print(
            fusion[
                [
                    "FeatureSet",
                    "C_index_Mean",
                    "C_index_SD",
                    "C_index_CI95_Low",
                    "C_index_CI95_High",
                    "FusionWeight_Mean",
                ]
            ].sort_values("C_index_Mean", ascending=False).to_string(index=False)
        )

    if not df_ranking.empty:
        print("\n[所有组合C-index排名 Top 15]")
        print(df_ranking.head(15).to_string(index=False))

    if not df_paired.empty:
        print("\n[QUBO_NoRed-Fusion 相对Fusion特征基线配对比较]")
        print(df_paired.to_string(index=False))

    if not df_target_advantage.empty:
        print("\n[QUBO_NoRed-Fusion 相对所有组合配对比较]")
        print(df_target_advantage.to_string(index=False))


if __name__ == "__main__":
    main()
