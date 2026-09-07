# -*- coding: utf-8 -*-
"""
方向二：QBM-inspired QUBO + RF/RSF + BP-Cox 融合预后模型

特点：
1. 不使用 VAE。
2. 默认只使用 OS 终点，避免 OS/DFS/RFS 混合。
3. 每个 cohort 内 z-score，降低批次效应和子集不纯影响。
4. QUBO 选择预后相关且低冗余基因。
5. 模型对比：
   - CoxPH
   - Lasso-Cox
   - PCA-Cox
   - RF/RSF
   - BP-Cox
   - RF/RSF + BP-Cox Fusion
6. 包含超参调优、交叉验证、消融实验、可视化。
"""

import os
import json
import time
import random
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

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

import torch
import torch.nn as nn
import torch.optim as optim


# =========================================================
# 0. 配置
# =========================================================

DATA_ROOT = params.DATA_ROOT
OUTPUT_ROOT = params.OUTPUT_MAIN


def parse_args(argv=None):
    """Parse CLI arguments; falls back to the defaults above when none are given."""
    parser = argparse.ArgumentParser(description="QUBO-Fusion core experiment: QUBO gene selection + RSF/BP-Cox/Fusion")
    parser.add_argument("--data_root", default=DATA_ROOT,
                        help="Root dir of raw data (mrna.rda files, one subdir per cancer)")
    parser.add_argument("--output", dest="output_root", default=OUTPUT_ROOT,
                        help="Root dir for all outputs")
    parser.add_argument("--cancers", default=",".join(CANCERS),
                        help="Comma-separated cancer list, default: all 6 cancers")
    return parser.parse_args(argv)


def apply_config(args):
    """Write CLI args back into module-level config (used by this script and external importlib callers)."""
    global DATA_ROOT, OUTPUT_ROOT, CANCERS
    DATA_ROOT = args.data_root
    OUTPUT_ROOT = args.output_root
    CANCERS = [c.strip() for c in args.cancers.split(",") if c.strip()]


CANCERS = params.CANCERS

RANDOM_SEED = params.RANDOM_SEED

MAX_UNIVARIATE_GENES = params.MAX_UNIVARIATE_GENES
MAX_MERGED_GENES = params.MAX_MERGED_GENES
MAX_QUBO_CANDIDATE_GENES = params.MAX_QUBO_CANDIDATE_GENES
SELECTED_GENE_NUM = params.SELECTED_GENE_NUM

TEST_SIZE = params.TEST_SIZE
CV_FOLDS = params.CV_FOLDS

MIN_SAMPLE_NUM = params.MIN_SAMPLE_NUM
MIN_GENE_NUM = params.MIN_GENE_NUM
MIN_SURVIVAL_TIME = params.MIN_SURVIVAL_TIME

ALLOW_MIXED_ENDPOINTS = params.ALLOW_MIXED_ENDPOINTS
TARGET_ENDPOINT = params.TARGET_ENDPOINT
MIN_COHORT_EVENT_NUM = params.MIN_COHORT_EVENT_NUM
USE_COHORT_ZSCORE = params.USE_COHORT_ZSCORE
ENDPOINT_FALLBACK_IF_TOO_SMALL = params.ENDPOINT_FALLBACK_IF_TOO_SMALL
MIN_TOTAL_SAMPLE_AFTER_ENDPOINT_FILTER = params.MIN_TOTAL_SAMPLE_AFTER_ENDPOINT_FILTER

QUBO_RELEVANCE_WEIGHT = params.QUBO_RELEVANCE_WEIGHT
QUBO_REDUNDANCY_WEIGHT = params.QUBO_REDUNDANCY_WEIGHT
QUBO_CARDINALITY_WEIGHT = params.QUBO_CARDINALITY_WEIGHT

USE_KAIWU_FIRST = params.USE_KAIWU_FIRST
SA_STEPS = params.SA_STEPS
SA_T0 = params.SA_T0
SA_T1 = params.SA_T1

PCA_COMPONENTS = params.PCA_COMPONENTS

RSF_PARAM_GRID = params.RSF_PARAM_GRID

RSF_N_JOBS = params.RSF_N_JOBS
RSF_PREDICT_CHUNK_SIZE = params.RSF_PREDICT_CHUNK_SIZE

FORCE_RF_FALLBACK = params.FORCE_RF_FALLBACK

BPCOX_PARAM_GRID = params.BPCOX_PARAM_GRID

FUSION_WEIGHT_GRID = params.FUSION_WEIGHT_GRID

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 1. 基础工具
# =========================================================

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_to_csv(df, path, **kwargs):
    ensure_dir(os.path.dirname(path))
    try:
        df.to_csv(path, **kwargs)
        return path
    except PermissionError:
        root, ext = os.path.splitext(path)
        alt_path = f"{root}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
        print(f"⚠️ 文件被占用，改存为: {alt_path}")
        df.to_csv(alt_path, **kwargs)
        return alt_path


def safe_name(x):
    return str(x).replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def normalize_col_name(x):
    return str(x).strip().lower().replace("-", "_").replace(".", "_").replace(" ", "_")


def clean_sample_id(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().replace('"', "").replace("'", "")
    if len(x) >= 12 and x[:4].upper() == "TCGA":
        return x[:12]
    return x


def cohort_zscore(expr):
    expr = expr.copy()
    mean = expr.mean(axis=0)
    std = expr.std(axis=0).replace(0, np.nan)
    expr = (expr - mean) / std
    expr = expr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return expr


set_seed(RANDOM_SEED)
setup_plot_style()
ensure_dir(OUTPUT_ROOT)


# =========================================================
# 2. 可选依赖
# =========================================================

SKSURV_AVAILABLE = False
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    SKSURV_AVAILABLE = True
except Exception:
    SKSURV_AVAILABLE = False

if FORCE_RF_FALLBACK:
    SKSURV_AVAILABLE = False


KAIWU_AVAILABLE = False
kw = None

try:
    import kaiwu as kw
    KAIWU_AVAILABLE = True
    print("Kaiwu SDK loaded:", getattr(kw, "__version__", "unknown"))
except Exception:
    try:
        import kaiwu_sdk as kw
        KAIWU_AVAILABLE = True
        print("kaiwu_sdk loaded")
    except Exception:
        KAIWU_AVAILABLE = False
        print("Kaiwu unavailable, fallback SA will be used.")


# =========================================================
# 3. RDA 读取
# =========================================================

def extract_dataframes_recursive(obj, prefix="obj"):
    dfs = {}

    if isinstance(obj, pd.DataFrame):
        dfs[prefix] = obj
    elif isinstance(obj, pd.Series):
        dfs[prefix] = obj.to_frame()
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 2:
            dfs[prefix] = pd.DataFrame(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            dfs.update(extract_dataframes_recursive(v, f"{prefix}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            dfs.update(extract_dataframes_recursive(v, f"{prefix}.{i}"))

    return dfs


def read_rda_as_dataframes(path):
    all_dfs = {}

    try:
        import pyreadr
        res = pyreadr.read_r(path)
        for k, v in res.items():
            all_dfs.update(extract_dataframes_recursive(v, k))
        if all_dfs:
            return all_dfs
    except Exception:
        pass

    try:
        import rdata
        parsed = rdata.parser.parse_file(path)
        converted = rdata.conversion.convert(parsed)
        all_dfs.update(extract_dataframes_recursive(converted, "rdata"))
        if all_dfs:
            return all_dfs
    except Exception as e:
        raise RuntimeError(f"读取 RDA 失败: {path}, error={e}")

    return all_dfs


# =========================================================
# 4. 临床表和表达表解析
# =========================================================

def is_clinical_table_from_mrna(name, df):
    lname = str(name).lower()
    if "clin" in lname or "surv" in lname:
        return True

    cols = [normalize_col_name(c) for c in df.columns]
    kws = ["os", "pfs", "dfs", "rfs", "dss", "time", "status", "event", "vital", "death"]
    return sum(any(k in c for k in kws) for c in cols) >= 2


def is_expression_table_from_mrna(name, df):
    lname = str(name).lower()
    if "expr" in lname or "exp" in lname or "mrna" in lname:
        return True

    if df.shape[0] >= 50 and df.shape[1] >= 10:
        ratio = df.apply(pd.to_numeric, errors="coerce").notna().mean().mean()
        return ratio > 0.5

    return False


def find_sample_id_col(df):
    cols = list(df.columns)
    norm = {c: normalize_col_name(c) for c in cols}

    exact = [
        "sample", "sample_id", "samples", "sampleid",
        "barcode", "patient", "patient_id", "patientid",
        "id", "geo_accession", "gsm", "case_id"
    ]

    for c in cols:
        if norm[c] in exact:
            return c

    for c in cols:
        nc = norm[c]
        if "sample" in nc or "barcode" in nc or "patient" in nc or "geo" in nc or "gsm" in nc or "case" in nc:
            return c

    best_col, best_score = None, 0

    for c in cols:
        vals = df[c].dropna().astype(str).head(200).tolist()
        score = 0
        for v in vals:
            u = v.upper()
            if u.startswith("TCGA") or u.startswith("GSM") or u.startswith("SRR") or u.startswith("ERR"):
                score += 1
        if score > best_score:
            best_col, best_score = c, score

    if best_score >= 3:
        return best_col

    if len(cols) > 0:
        c = cols[0]
        vals = df[c].dropna().astype(str).head(50)
        if len(vals) > 0:
            num_ratio = pd.to_numeric(vals, errors="coerce").notna().mean()
            uniq_ratio = vals.nunique() / max(1, len(vals))
            if num_ratio < 0.5 and uniq_ratio > 0.8:
                return c

    return None


def normalize_event_value(x):
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

    if s.startswith("1:") or s.startswith("1_"):
        return 1
    if s.startswith("0:") or s.startswith("0_"):
        return 0

    if "dead" in s or "deceased" in s or "death" in s or "relapse" in s or "progress" in s or "recur" in s:
        return 1
    if "alive" in s or "censor" in s or "living" in s or "disease_free" in s:
        return 0

    try:
        v = float(s)
        return 1 if v > 0 else 0
    except Exception:
        return np.nan


def pick_survival_columns(df):
    cols = list(df.columns)
    norm = {c: normalize_col_name(c) for c in cols}
    endpoints = ["os", "pfs", "dfs", "rfs", "dss", "dmfs", "drfs"]

    for ep in endpoints:
        tcs, ecs = [], []

        for c in cols:
            nc = norm[c]

            if (
                nc == f"{ep}_time"
                or nc == f"{ep}.time"
                or nc == f"{ep}time"
                or (
                    ep in nc and any(k in nc for k in [
                        "time", "days", "day", "month", "months",
                        "year", "years", "survival", "follow", "followup"
                    ])
                )
            ):
                tcs.append(c)

            if (
                nc == ep
                or nc == f"{ep}_status"
                or nc == f"{ep}.status"
                or nc == f"{ep}_event"
                or nc == f"{ep}.event"
                or (
                    ep in nc and any(k in nc for k in [
                        "status", "event", "censor", "death", "dead",
                        "recurrence", "progression", "relapse"
                    ])
                )
            ):
                ecs.append(c)

        if tcs and ecs:
            return tcs[0], ecs[0], ep.upper()

    time_keywords = [
        "time", "days", "day", "month", "months",
        "year", "years", "futime", "survival",
        "follow", "followup", "follow_up"
    ]

    event_keywords = [
        "status", "event", "censor", "death", "dead",
        "vital", "recurrence", "progression", "relapse",
        "os", "pfs", "dfs", "rfs", "dss", "dmfs", "drfs"
    ]

    tcs, ecs = [], []

    for c in cols:
        nc = norm[c]
        if any(k in nc for k in time_keywords):
            tcs.append(c)
        if any(k in nc for k in event_keywords):
            ecs.append(c)

    ecs = [
        c for c in ecs
        if not (
            "sample" in norm[c]
            or "patient" in norm[c]
            or "barcode" in norm[c]
            or norm[c] in ["id", "sample_id", "patient_id"]
        )
    ]

    for tc in tcs:
        for ec in ecs:
            if tc != ec:
                endpoint = "UNKNOWN"
                tc_l = normalize_col_name(tc)
                ec_l = normalize_col_name(ec)

                for ep in endpoints:
                    if ep in tc_l or ep in ec_l:
                        endpoint = ep.upper()
                        break

                return tc, ec, endpoint

    numeric_cols = []
    event_cols = []

    for c in cols:
        nc = norm[c]

        if (
            "sample" in nc
            or "patient" in nc
            or "barcode" in nc
            or nc in ["id", "sample_id", "patient_id"]
        ):
            continue

        num = pd.to_numeric(df[c], errors="coerce")

        if num.notna().mean() >= 0.5:
            vals = num.dropna()
            if len(vals) >= 5:
                binary_like = set(vals.unique()).issubset({0, 1, 0.0, 1.0})
                if (
                    (vals > MIN_SURVIVAL_TIME).mean() >= 0.5
                    and vals.nunique() >= 5
                    and not binary_like
                ):
                    numeric_cols.append(c)

        ev = df[c].map(normalize_event_value)

        if ev.notna().mean() >= 0.5:
            evv = ev.dropna().astype(int)
            if len(evv) >= 5 and evv.nunique() >= 2 and evv.sum() >= 1:
                event_cols.append(c)

    if numeric_cols and event_cols:
        for tc in numeric_cols:
            for ec in event_cols:
                if tc != ec:
                    endpoint = "INFERRED"
                    tc_l = normalize_col_name(tc)
                    ec_l = normalize_col_name(ec)

                    for ep in endpoints:
                        if ep in tc_l or ep in ec_l:
                            endpoint = ep.upper()
                            break

                    return tc, ec, endpoint

    return None, None, None


def prepare_clinical_table_from_mrna(df, cohort_name):
    if df is None or not isinstance(df, pd.DataFrame):
        return None

    raw = df.copy()
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if raw.shape[0] == 0 or raw.shape[1] == 0:
        return None

    candidate_tables = [("raw", raw.copy())]

    try:
        col_names = [str(c) for c in raw.columns]
        numeric_like = all(c.isdigit() for c in col_names)
        v_like = all(c.lower().startswith("v") and c[1:].isdigit() for c in col_names if len(c) > 1)

        if numeric_like or v_like:
            promoted = raw.copy()
            promoted.columns = promoted.iloc[0].astype(str).tolist()
            promoted = promoted.iloc[1:].copy()
            candidate_tables.append(("first_row_as_header", promoted))
    except Exception:
        pass

    try:
        trans = raw.T.copy()
        trans.columns = [str(c) for c in trans.columns]
        candidate_tables.append(("transposed", trans))

        trans2 = raw.T.copy()
        trans2.columns = trans2.iloc[0].astype(str).tolist()
        trans2 = trans2.iloc[1:].copy()
        candidate_tables.append(("transposed_first_row_as_header", trans2))
    except Exception:
        pass

    def unique_columns(tab):
        tab = tab.copy()
        tab.columns = [str(c) for c in tab.columns]

        if len(set(tab.columns)) < len(tab.columns):
            seen = {}
            new_cols = []

            for c in tab.columns:
                if c not in seen:
                    seen[c] = 0
                    new_cols.append(c)
                else:
                    seen[c] += 1
                    new_cols.append(f"{c}_{seen[c]}")

            tab.columns = new_cols

        return tab

    def try_parse(tab, tag):
        tab = tab.copy()
        tab = tab.dropna(axis=0, how="all").dropna(axis=1, how="all")
        tab = unique_columns(tab)

        if tab.shape[0] == 0 or tab.shape[1] == 0:
            return None

        sid_col = find_sample_id_col(tab)

        idx_vals = [str(x) for x in tab.index]
        tcga_like = sum(x.upper().startswith("TCGA") for x in idx_vals)
        geo_like = sum(("GSM" in x.upper() or "SRR" in x.upper() or "ERR" in x.upper()) for x in idx_vals)
        index_as_sample = (tcga_like + geo_like) >= max(2, int(len(idx_vals) * 0.2))

        if sid_col is not None:
            sample_ids = tab[sid_col].map(clean_sample_id).astype(str)
        elif index_as_sample:
            sample_ids = pd.Series(tab.index, index=tab.index).map(clean_sample_id).astype(str)
        else:
            sample_ids = pd.Series(tab.index, index=tab.index).map(clean_sample_id).astype(str)

        time_col, event_col, endpoint = pick_survival_columns(tab)

        if time_col is None or event_col is None:
            return None

        out = pd.DataFrame()
        out["sample_id"] = list(sample_ids)
        out["time"] = pd.to_numeric(tab[time_col].values, errors="coerce")
        out["event"] = pd.Series(tab[event_col].values).map(normalize_event_value).values

        out = out.dropna(subset=["sample_id", "time", "event"])
        out = out[out["time"] > MIN_SURVIVAL_TIME]

        if out.shape[0] < 5:
            return None

        out["event"] = out["event"].astype(int)

        if out["event"].sum() < 1:
            return None

        out["cohort"] = cohort_name
        out["endpoint"] = endpoint
        out["clinical_orientation"] = tag
        out["time_col"] = str(time_col)
        out["event_col"] = str(event_col)

        out = out.drop_duplicates("sample_id").set_index("sample_id")

        out.attrs["clinical_columns"] = [str(c) for c in tab.columns]
        out.attrs["time_col"] = str(time_col)
        out.attrs["event_col"] = str(event_col)
        out.attrs["orientation"] = str(tag)

        return out

    parsed = []

    for tag, tab in candidate_tables:
        res = try_parse(tab, tag)
        if res is not None:
            parsed.append(res)

    if not parsed:
        return None

    parsed = sorted(parsed, key=lambda x: (x.shape[0], x["event"].sum()), reverse=True)

    return parsed[0]


def prepare_expression_table_from_mrna(df, clinical_ids):
    if df is None or not isinstance(df, pd.DataFrame):
        return None

    raw = df.copy()
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if raw.shape[0] == 0 or raw.shape[1] == 0:
        return None

    clinical_ids = set(map(clean_sample_id, clinical_ids))

    def count_overlap(values):
        return sum(clean_sample_id(x) in clinical_ids for x in values)

    candidates = []

    try:
        ov = count_overlap(raw.columns)
        if ov > 0:
            expr = raw.T.copy()
            expr.index = [clean_sample_id(x) for x in expr.index]
            expr.columns = [str(x) for x in raw.index]
            candidates.append(("columns_are_samples", ov, expr))
    except Exception:
        pass

    try:
        ov = count_overlap(raw.index)
        if ov > 0:
            expr = raw.copy()
            expr.index = [clean_sample_id(x) for x in expr.index]
            expr.columns = [str(x) for x in expr.columns]
            candidates.append(("rows_are_samples", ov, expr))
    except Exception:
        pass

    try:
        first_col = raw.columns[0]
        ov = count_overlap(raw[first_col].astype(str).values)
        if ov > 0:
            expr = raw.copy()
            expr.index = [clean_sample_id(x) for x in expr[first_col].astype(str).values]
            expr = expr.drop(columns=[first_col])
            candidates.append(("first_column_is_sample_id", ov, expr))
    except Exception:
        pass

    try:
        first_col = raw.columns[0]
        ov = count_overlap(list(raw.columns[1:]))
        if ov > 0:
            expr = raw.copy()
            genes = expr[first_col].astype(str).values
            expr = expr.drop(columns=[first_col]).T.copy()
            expr.index = [clean_sample_id(x) for x in expr.index]
            expr.columns = genes
            candidates.append(("first_column_is_gene_columns_are_samples", ov, expr))
    except Exception:
        pass

    if not candidates:
        return None

    orientation, overlap, expr = sorted(candidates, key=lambda x: x[1], reverse=True)[0]

    expr = expr.loc[[i for i in expr.index if i in clinical_ids]]

    if expr.shape[0] == 0:
        return None

    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(axis=1, how="all")

    if expr.shape[1] == 0:
        return None

    expr = expr.loc[:, expr.notna().mean(axis=0) >= 0.5]

    if expr.shape[1] == 0:
        return None

    expr = expr.fillna(expr.median(axis=0))
    expr = expr.loc[:, expr.var(axis=0) > 1e-10]

    if expr.shape[0] == 0 or expr.shape[1] == 0:
        return None

    try:
        max_val = np.nanmax(expr.values)
        min_val = np.nanmin(expr.values)

        if max_val > 50 and min_val >= 0:
            expr = np.log2(expr + 1.0)
    except Exception:
        pass

    expr = expr.groupby(expr.index).mean()
    expr = expr.T.groupby(expr.columns).mean().T

    expr.attrs["orientation"] = orientation
    expr.attrs["overlap"] = int(overlap)

    return expr


def pair_mrna_tables(dfs):
    clinical = {}
    expression = {}

    for name, df in dfs.items():
        if not isinstance(df, pd.DataFrame):
            continue

        if is_clinical_table_from_mrna(name, df):
            clinical[name] = df
        elif is_expression_table_from_mrna(name, df):
            expression[name] = df

    return clinical, expression


def get_cohort_name_from_key(name):
    return str(name).split(".")[-1]


def load_cancer_data(cancer):
    print_section(f"加载癌种数据: {cancer}")

    rda_path = os.path.join(DATA_ROOT, cancer, "mrna.rda")

    if not os.path.exists(rda_path):
        raise FileNotFoundError(f"找不到 mrna.rda: {rda_path}")

    dfs = read_rda_as_dataframes(rda_path)
    clin_raw, expr_raw = pair_mrna_tables(dfs)

    print("clinical candidates:", list(clin_raw.keys())[:10])
    print("expression candidates:", list(expr_raw.keys())[:10])

    expr_by_cohort = {
        get_cohort_name_from_key(k): (k, v)
        for k, v in expr_raw.items()
    }

    expr_list = []
    surv_list = []
    cohort_infos = []

    fallback_expr_list = []
    fallback_surv_list = []
    fallback_cohort_infos = []

    debug_rows = []

    for clin_name, clin_df in clin_raw.items():
        cohort = get_cohort_name_from_key(clin_name)

        debug = {
            "cohort": cohort,
            "clinical_table": clin_name,
            "clinical_shape": str(clin_df.shape),
            "status": "",
            "reason": "",
            "surv_n": None,
            "event_n": None,
            "expr_n": None,
            "gene_n": None,
            "overlap": None,
            "endpoint": None,
            "target_endpoint": TARGET_ENDPOINT,
            "expr_table": None,
            "time_col": None,
            "event_col": None,
            "clinical_orientation": None,
            "expression_orientation": None,
            "cohort_zscore": USE_COHORT_ZSCORE,
        }

        try:
            surv = prepare_clinical_table_from_mrna(clin_df, cohort)

            if surv is None:
                debug["status"] = "failed"
                debug["reason"] = "clinical_parse_failed"
                debug_rows.append(debug)
                continue

            endpoint_val = str(surv["endpoint"].iloc[0]).upper()

            debug["surv_n"] = int(surv.shape[0])
            debug["event_n"] = int(surv["event"].sum())
            debug["endpoint"] = endpoint_val
            debug["time_col"] = str(surv["time_col"].iloc[0]) if "time_col" in surv.columns else ""
            debug["event_col"] = str(surv["event_col"].iloc[0]) if "event_col" in surv.columns else ""
            debug["clinical_orientation"] = (
                str(surv["clinical_orientation"].iloc[0])
                if "clinical_orientation" in surv.columns
                else ""
            )

            if surv.shape[0] < MIN_SAMPLE_NUM:
                debug["status"] = "failed"
                debug["reason"] = f"clinical_sample_too_small:{surv.shape[0]}"
                debug_rows.append(debug)
                continue

            if surv["event"].sum() < MIN_COHORT_EVENT_NUM:
                debug["status"] = "failed"
                debug["reason"] = f"event_too_small:{int(surv['event'].sum())}"
                debug_rows.append(debug)
                continue

            if cohort in expr_by_cohort:
                candidate_expr_tables = [expr_by_cohort[cohort]]
            else:
                candidate_expr_tables = list(expr_raw.items())

            best_expr = None
            best_name = None
            best_overlap = 0

            for expr_name, expr_df in candidate_expr_tables:
                expr = prepare_expression_table_from_mrna(expr_df, surv.index)

                if expr is None:
                    continue

                overlap = len(set(expr.index) & set(surv.index))

                if overlap > best_overlap:
                    best_expr = expr
                    best_name = expr_name
                    best_overlap = overlap

            debug["expr_table"] = best_name
            debug["overlap"] = int(best_overlap)

            if best_expr is None:
                debug["status"] = "failed"
                debug["reason"] = "expression_parse_failed_or_no_overlap"
                debug_rows.append(debug)
                continue

            if best_overlap < MIN_SAMPLE_NUM:
                debug["status"] = "failed"
                debug["reason"] = f"overlap_too_small:{best_overlap}"
                debug_rows.append(debug)
                continue

            common = sorted(list(set(best_expr.index) & set(surv.index)))

            expr = best_expr.loc[common].copy()
            surv2 = surv.loc[common].copy()

            debug["expr_n"] = int(expr.shape[0])
            debug["gene_n"] = int(expr.shape[1])
            debug["expression_orientation"] = str(best_expr.attrs.get("orientation", ""))

            if expr.shape[1] < MIN_GENE_NUM:
                debug["status"] = "failed"
                debug["reason"] = f"gene_too_small:{expr.shape[1]}"
                debug_rows.append(debug)
                continue

            if USE_COHORT_ZSCORE:
                expr = cohort_zscore(expr)

            new_idx = [f"{cohort}__{x}" for x in common]
            expr.index = new_idx
            surv2.index = new_idx

            cohort_info = {
                "cohort": cohort,
                "clinical_table": clin_name,
                "expression_table": best_name,
                "sample_num": int(expr.shape[0]),
                "gene_num": int(expr.shape[1]),
                "event_num": int(surv2["event"].sum()),
                "endpoint": endpoint_val,
                "time_col": debug["time_col"],
                "event_col": debug["event_col"],
                "clinical_orientation": debug["clinical_orientation"],
                "expression_orientation": debug["expression_orientation"],
                "cohort_zscore": USE_COHORT_ZSCORE,
            }

            fallback_expr_list.append(expr)
            fallback_surv_list.append(surv2)
            fallback_cohort_infos.append(cohort_info)

            if not ALLOW_MIXED_ENDPOINTS and endpoint_val != str(TARGET_ENDPOINT).upper():
                debug["status"] = "failed"
                debug["reason"] = f"endpoint_filtered:{endpoint_val}"
                debug_rows.append(debug)
                print(f"  ⚠️ {cohort}: endpoint={endpoint_val} 被过滤，目标 endpoint={TARGET_ENDPOINT}")
                continue

            expr_list.append(expr)
            surv_list.append(surv2)
            cohort_infos.append(cohort_info)

            debug["status"] = "success"
            debug["reason"] = "ok"
            debug_rows.append(debug)

            print(
                f"  ✅ {cohort}: samples={expr.shape[0]}, genes={expr.shape[1]}, "
                f"events={int(surv2['event'].sum())}, endpoint={endpoint_val}, "
                f"time_col={debug['time_col']}, event_col={debug['event_col']}"
            )

        except Exception as e:
            debug["status"] = "failed"
            debug["reason"] = f"exception:{str(e)}"
            debug_rows.append(debug)
            traceback.print_exc()

    out_dir = os.path.join(OUTPUT_ROOT, cancer)
    ensure_dir(out_dir)

    debug_df = pd.DataFrame(debug_rows)
    debug_path = os.path.join(out_dir, f"{cancer}_cohort_loading_debug.csv")
    safe_to_csv(debug_df, debug_path, index=False, encoding="utf-8-sig")

    temp_sample_n = sum(x.shape[0] for x in expr_list) if expr_list else 0

    if (
        ENDPOINT_FALLBACK_IF_TOO_SMALL
        and temp_sample_n < MIN_TOTAL_SAMPLE_AFTER_ENDPOINT_FILTER
        and fallback_expr_list
    ):
        print(f"⚠️ {cancer} endpoint={TARGET_ENDPOINT} 样本数 {temp_sample_n} 太少，启用 mixed endpoint fallback")
        expr_list = fallback_expr_list
        surv_list = fallback_surv_list
        cohort_infos = fallback_cohort_infos

    if not expr_list:
        print("\n没有成功 cohort。失败原因统计：")
        if len(debug_df):
            print(debug_df["reason"].value_counts())
            print(debug_df.head(20))
        raise RuntimeError(f"{cancer} 没有可用 cohort，请查看 {debug_path}")

    expr = pd.concat(expr_list, axis=0, join="outer", copy=False)
    surv = pd.concat(surv_list, axis=0)

    expr = expr.loc[surv.index]
    expr = expr.loc[:, expr.notna().mean(axis=0) >= 0.6]

    if expr.shape[1] > MAX_MERGED_GENES:
        var = expr.var(axis=0, skipna=True).sort_values(ascending=False)
        expr = expr[var.index[:MAX_MERGED_GENES]]

    expr = expr.fillna(expr.median(axis=0))
    expr = expr.loc[:, expr.var(axis=0) > 1e-10]

    info = {
        "cancer": cancer,
        "sample_num": int(expr.shape[0]),
        "gene_num": int(expr.shape[1]),
        "event_num": int(surv["event"].sum()),
        "endpoint_counts": surv["endpoint"].value_counts().to_dict(),
        "target_endpoint": TARGET_ENDPOINT,
        "allow_mixed_endpoints": ALLOW_MIXED_ENDPOINTS,
        "use_cohort_zscore": USE_COHORT_ZSCORE,
        "cohorts": cohort_infos,
    }

    print("merged expr:", expr.shape)
    print("merged surv:", surv.shape)
    print("events:", int(surv["event"].sum()))
    print("endpoint_counts:", surv["endpoint"].value_counts().to_dict())
    print("successful cohorts:", len(cohort_infos))

    return expr, surv, info


# =========================================================
# 5. 单基因 Cox
# =========================================================

def fit_cox_one_gene(gene, x, surv):
    df = pd.DataFrame({
        "time": surv["time"].values,
        "event": surv["event"].values,
        gene: x
    })

    try:
        cph = CoxPHFitter(penalizer=0.05)
        cph.fit(df, duration_col="time", event_col="event")

        row = cph.summary.loc[gene]
        risk = cph.predict_partial_hazard(df[[gene]]).values.reshape(-1)
        cidx = concordance_index(surv["time"], -risk, surv["event"])

        return {
            "gene": gene,
            "coef": float(row["coef"]),
            "z": float(row["z"]),
            "p": float(row["p"]),
            "cindex": float(cidx),
            "variance": float(np.var(x))
        }
    except Exception:
        return None


def univariate_cox_scores(expr, surv, max_genes=2000):
    print_section("训练集单基因 Cox 打分")

    variances = expr.var(axis=0).sort_values(ascending=False)
    genes = list(variances.index[:min(max_genes, len(variances))])

    rows = []

    for i, gene in enumerate(genes):
        res = fit_cox_one_gene(gene, expr[gene].values, surv)

        if res is not None:
            rows.append(res)

        if (i + 1) % 200 == 0:
            print(f"processed {i + 1}/{len(genes)}, valid={len(rows)}")

    if not rows:
        raise RuntimeError("单基因 Cox 没有有效结果")

    df = pd.DataFrame(rows)
    df["abs_z"] = df["z"].abs()
    df["abs_z_rank"] = df["abs_z"].rank(pct=True)
    df["cindex_rank"] = df["cindex"].rank(pct=True)
    df["variance_rank"] = df["variance"].rank(pct=True)

    df["rank_score"] = (
        0.60 * df["abs_z_rank"] +
        0.30 * df["cindex_rank"] +
        0.10 * df["variance_rank"]
    )

    return df.sort_values("rank_score", ascending=False).reset_index(drop=True)


# =========================================================
# 6. QUBO
# =========================================================

def add_qubo_term(qubo, a, b, value):
    if a > b:
        a, b = b, a
    qubo[(a, b)] = qubo.get((a, b), 0.0) + float(value)


def build_gene_selection_qubo(expr_train, score_df, select_k, use_redundancy=True, redundancy_weight=0.35):
    cand = score_df.head(min(MAX_QUBO_CANDIDATE_GENES, len(score_df))).copy()
    genes = cand["gene"].tolist()

    rel = cand["rank_score"].values.astype(float)
    rel = rel / (np.max(rel) + 1e-8)

    X = expr_train[genes].values.astype(float)

    if use_redundancy and len(genes) > 1:
        corr = np.corrcoef(X.T)
        red = np.abs(np.nan_to_num(corr))
        np.fill_diagonal(red, 0)
    else:
        red = np.zeros((len(genes), len(genes)))

    n = len(genes)
    k = min(select_k, n)
    qubo = {}

    for i in range(n):
        add_qubo_term(qubo, f"x_{i}", f"x_{i}", -QUBO_RELEVANCE_WEIGHT * rel[i])

    for i in range(n):
        for j in range(i + 1, n):
            add_qubo_term(qubo, f"x_{i}", f"x_{j}", redundancy_weight * red[i, j])

    for i in range(n):
        add_qubo_term(qubo, f"x_{i}", f"x_{i}", QUBO_CARDINALITY_WEIGHT * (1 - 2 * k))

    for i in range(n):
        for j in range(i + 1, n):
            add_qubo_term(qubo, f"x_{i}", f"x_{j}", 2 * QUBO_CARDINALITY_WEIGHT)

    meta = {
        "candidate_genes": genes,
        "select_k": int(k),
        "candidate_n": int(n),
        "use_redundancy": bool(use_redundancy),
        "redundancy_weight": float(redundancy_weight)
    }

    return qubo, meta, cand


def qubo_energy(qubo, state):
    return float(
        sum(
            float(w) * int(state.get(a, 0)) * int(state.get(b, 0))
            for (a, b), w in qubo.items()
        )
    )


def get_qubo_variables(qubo):
    return sorted(
        list(set([v for pair in qubo.keys() for v in pair])),
        key=lambda x: int(str(x).split("_")[-1])
    )


def solve_qubo_by_classical_sa(qubo):
    vars_ = get_qubo_variables(qubo)
    state = {v: random.randint(0, 1) for v in vars_}
    e = qubo_energy(qubo, state)
    best, best_e = dict(state), e

    for step in range(SA_STEPS):
        frac = step / max(SA_STEPS - 1, 1)
        temp = SA_T0 * ((SA_T1 / SA_T0) ** frac)

        v = random.choice(vars_)
        ns = dict(state)
        ns[v] = 1 - ns[v]
        ne = qubo_energy(qubo, ns)
        delta = ne - e

        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-12)):
            state, e = ns, ne
            if e < best_e:
                best, best_e = dict(state), e

    return best, best_e, {"solver": "internal_sa", "energy": float(best_e), "fallback_used": True}


def qubo_dict_to_kaiwu_model(qubo):
    variables = get_qubo_variables(qubo)
    bin_vars = {v: kw.core.Binary(v) for v in variables}

    model = kw.qubo.QuboModel()
    obj = 0

    for (a, b), w in qubo.items():
        if a == b:
            obj += float(w) * bin_vars[a]
        else:
            obj += float(w) * bin_vars[a] * bin_vars[b]

    model.set_objective(obj)
    return model, variables


def normalize_kaiwu_solution(sol, variables):
    raw = sol[0] if isinstance(sol, tuple) else getattr(sol, "best_solution", getattr(sol, "solution", sol))

    if isinstance(raw, dict):
        return {
            v: int(float(raw.get(v, raw.get(i, raw.get(str(i), 0)))) >= 0.5)
            for i, v in enumerate(variables)
        }

    arr = np.asarray(raw).reshape(-1)

    return {
        v: int(float(arr[i]) >= 0.5) if i < len(arr) else 0
        for i, v in enumerate(variables)
    }


def solve_qubo_by_kaiwu(qubo):
    model, variables = qubo_dict_to_kaiwu_model(qubo)
    optimizer = kw.classical.SimulatedAnnealingOptimizer()
    solver = kw.solver.SimpleSolver(optimizer)
    sol = solver.solve_qubo(model)

    state = normalize_kaiwu_solution(sol, variables)
    e = qubo_energy(qubo, state)

    return state, e, {"solver": "kaiwu_sa", "energy": float(e), "fallback_used": False}


def repair_solution_to_k(state, qubo, n, k):
    state = {f"x_{i}": int(state.get(f"x_{i}", 0)) for i in range(n)}

    while sum(state.values()) > k:
        best, best_e = None, None
        for v in list(state.keys()):
            if state[v] == 1:
                temp = dict(state)
                temp[v] = 0
                e = qubo_energy(qubo, temp)
                if best_e is None or e < best_e:
                    best, best_e = temp, e
        state = best

    while sum(state.values()) < k:
        best, best_e = None, None
        for v in list(state.keys()):
            if state[v] == 0:
                temp = dict(state)
                temp[v] = 1
                e = qubo_energy(qubo, temp)
                if best_e is None or e < best_e:
                    best, best_e = temp, e
        state = best

    return state


def select_genes_by_qubo(expr_train, score_df, out_dir, tag="qubo_full", use_redundancy=True, redundancy_weight=0.35):
    print_section(f"QUBO 选择基因: {tag}")

    qubo, meta, cand = build_gene_selection_qubo(
        expr_train,
        score_df,
        SELECTED_GENE_NUM,
        use_redundancy,
        redundancy_weight
    )

    if USE_KAIWU_FIRST and KAIWU_AVAILABLE:
        try:
            state, energy, info = solve_qubo_by_kaiwu(qubo)
        except Exception:
            traceback.print_exc()
            state, energy, info = solve_qubo_by_classical_sa(qubo)
    else:
        state, energy, info = solve_qubo_by_classical_sa(qubo)

    repaired = repair_solution_to_k(state, qubo, meta["candidate_n"], meta["select_k"])

    selected_local = [
        i for i in range(meta["candidate_n"])
        if repaired.get(f"x_{i}", 0) == 1
    ]

    genes = [meta["candidate_genes"][i] for i in selected_local]

    selected_df = cand[cand["gene"].isin(genes)].copy()
    selected_df["selected_rank"] = range(1, len(selected_df) + 1)

    save_json(meta, os.path.join(out_dir, f"{tag}_qubo_meta.json"))
    save_json(info, os.path.join(out_dir, f"{tag}_solver_info.json"))

    save_json(
        {
            "raw_energy": float(energy),
            "repaired_energy": float(qubo_energy(qubo, repaired)),
            "selected_gene_num": int(len(genes))
        },
        os.path.join(out_dir, f"{tag}_repair_info.json")
    )

    safe_to_csv(
        selected_df,
        os.path.join(out_dir, f"{tag}_selected_genes.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("selected genes:", genes)

    return genes, selected_df


# =========================================================
# 7. 生存模型和调参
# =========================================================

def calc_cindex(surv, risk):
    return float(
        concordance_index(
            surv["time"].values,
            -np.asarray(risk).reshape(-1),
            surv["event"].values
        )
    )


def calc_logrank_hr(surv, risk):
    risk = np.asarray(risk).reshape(-1)
    high = risk >= np.median(risk)
    low = ~high

    if high.sum() < 2 or low.sum() < 2:
        return np.nan, np.nan

    lr = logrank_test(
        surv.loc[high, "time"],
        surv.loc[low, "time"],
        event_observed_A=surv.loc[high, "event"],
        event_observed_B=surv.loc[low, "event"]
    )

    df = pd.DataFrame({
        "time": surv["time"].values,
        "event": surv["event"].values,
        "risk_group": high.astype(int)
    })

    try:
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(df, duration_col="time", event_col="event")
        hr = float(np.exp(cph.params_["risk_group"]))
    except Exception:
        hr = np.nan

    return float(lr.p_value), hr


def evaluate_survival_model(name, surv, risk):
    cidx = calc_cindex(surv, risk)
    pval, hr = calc_logrank_hr(surv, risk)

    return {
        "Model": name,
        "C_index": cidx,
        "logrank_p": pval,
        "HR_high_vs_low": hr
    }


def make_cv_splits(surv, n_splits=3):
    n = len(surv)
    n_splits = min(n_splits, max(2, n))
    event = surv["event"].values.astype(int)
    counts = np.bincount(event)

    if len(counts) >= 2 and np.min(counts) >= n_splits:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        return list(cv.split(np.zeros(n), event))

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    return list(cv.split(np.zeros(n)))


def fit_cox_predict(X_train, surv_train, X_test, penalizer=0.05, l1_ratio=0.0):
    df = pd.DataFrame(X_train, columns=[f"x{i}" for i in range(X_train.shape[1])])
    df["time"] = surv_train["time"].values
    df["event"] = surv_train["event"].values

    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(df, duration_col="time", event_col="event")

    cols = [c for c in df.columns if c not in ["time", "event"]]

    risk_train = cph.predict_partial_hazard(df[cols]).values.reshape(-1)
    risk_test = cph.predict_partial_hazard(pd.DataFrame(X_test, columns=cols)).values.reshape(-1)

    return cph, risk_train, risk_test


def cv_tune_cox(X, surv, l1_ratio=0.0):
    grid = [0.001, 0.01, 0.05, 0.1, 0.5]
    folds = make_cv_splits(surv, CV_FOLDS)
    best = None

    for pen in grid:
        vals = []
        for tr, va in folds:
            try:
                _, _, r = fit_cox_predict(X[tr], surv.iloc[tr], X[va], penalizer=pen, l1_ratio=l1_ratio)
                vals.append(calc_cindex(surv.iloc[va], r))
            except Exception:
                pass

        score = float(np.mean(vals)) if vals else -np.inf

        if best is None or score > best["cv_cindex"]:
            best = {"penalizer": pen, "l1_ratio": l1_ratio, "cv_cindex": score}

    return best


def to_sksurv_y(surv):
    return Surv.from_arrays(
        event=surv["event"].astype(bool).values,
        time=surv["time"].astype(float).values
    )


def rsf_predict_in_chunks(model, X, chunk_size=128):
    risks = []
    for start in range(0, X.shape[0], chunk_size):
        end = min(start + chunk_size, X.shape[0])
        pred = model.predict(X[start:end])
        risks.append(np.asarray(pred).reshape(-1))
    return np.concatenate(risks, axis=0)


def fit_rsf_predict(X_train, surv_train, X_test, params):
    if SKSURV_AVAILABLE:
        try:
            model = RandomSurvivalForest(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=RANDOM_SEED,
                n_jobs=RSF_N_JOBS
            )
            model.fit(X_train, to_sksurv_y(surv_train))

            risk_train = rsf_predict_in_chunks(model, X_train, RSF_PREDICT_CHUNK_SIZE)
            risk_test = rsf_predict_in_chunks(model, X_test, RSF_PREDICT_CHUNK_SIZE)

            return model, np.asarray(risk_train), np.asarray(risk_test)

        except Exception as e:
            print("⚠️ sksurv RSF 失败，切换到 RFRegressor fallback")
            print("error:", str(e))

    target = -np.log1p(surv_train["time"].values)
    sample_weight = np.where(surv_train["event"].values == 1, 1.5, 0.8)

    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=RANDOM_SEED,
        n_jobs=1
    )

    model.fit(X_train, target, sample_weight=sample_weight)

    risk_train = model.predict(X_train)
    risk_test = model.predict(X_test)

    return model, np.asarray(risk_train), np.asarray(risk_test)


def cv_tune_rsf(X, surv):
    folds = make_cv_splits(surv, CV_FOLDS)
    best = None

    for params in RSF_PARAM_GRID:
        vals = []
        for tr, va in folds:
            try:
                _, _, r = fit_rsf_predict(X[tr], surv.iloc[tr], X[va], params)
                vals.append(calc_cindex(surv.iloc[va], r))
            except Exception:
                pass

        score = float(np.mean(vals)) if vals else -np.inf
        item = dict(params)
        item["cv_cindex"] = score

        if best is None or score > best["cv_cindex"]:
            best = item

    return best


class BPCoxNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.25):
        super().__init__()
        h2 = max(4, hidden_dim // 2)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, h2),
            nn.LayerNorm(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1)
        )

    def forward(self, x):
        return self.net(x).reshape(-1)


def cox_ph_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    return -torch.sum((risk - log_cumsum) * event) / (torch.sum(event) + 1e-8)


def fit_bpcox_predict(X_train, surv_train, X_test, params):
    set_seed(RANDOM_SEED)

    model = BPCoxNet(
        X_train.shape[1],
        params["hidden_dim"],
        params["dropout"]
    ).to(DEVICE)

    opt = optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )

    x = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    t = torch.tensor(surv_train["time"].values.astype(float), dtype=torch.float32, device=DEVICE)
    e = torch.tensor(surv_train["event"].values.astype(float), dtype=torch.float32, device=DEVICE)

    best_loss, best_state, no_imp = None, None, 0

    model.train()

    for epoch in range(params["epochs"]):
        opt.zero_grad()
        risk = model(x)
        loss = cox_ph_loss(risk, t, e)
        loss.backward()
        opt.step()

        lv = float(loss.detach().cpu().item())

        if best_loss is None or lv < best_loss - 1e-5:
            best_loss = lv
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if no_imp >= 25:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():
        rtr = model(torch.tensor(X_train, dtype=torch.float32, device=DEVICE)).cpu().numpy()
        rte = model(torch.tensor(X_test, dtype=torch.float32, device=DEVICE)).cpu().numpy()

    return model, rtr, rte


def cv_tune_bpcox(X, surv):
    folds = make_cv_splits(surv, CV_FOLDS)
    best = None

    for params in BPCOX_PARAM_GRID:
        vals = []
        for tr, va in folds:
            try:
                _, _, r = fit_bpcox_predict(X[tr], surv.iloc[tr], X[va], params)
                vals.append(calc_cindex(surv.iloc[va], r))
            except Exception:
                pass

        score = float(np.mean(vals)) if vals else -np.inf
        item = dict(params)
        item["cv_cindex"] = score

        if best is None or score > best["cv_cindex"]:
            best = item

    return best


def standardize_risk_by_train(train_risk, test_risk):
    mu = np.mean(train_risk)
    sd = np.std(train_risk) + 1e-8
    return (train_risk - mu) / sd, (test_risk - mu) / sd


def tune_fusion_weight_cv(X, surv, rsf_params, bp_params):
    folds = make_cv_splits(surv, CV_FOLDS)
    best_w, best_score = 0.5, -np.inf
    rows = []

    for w in FUSION_WEIGHT_GRID:
        vals = []

        for tr, va in folds:
            try:
                _, rsf_tr, rsf_va = fit_rsf_predict(X[tr], surv.iloc[tr], X[va], rsf_params)
                _, bp_tr, bp_va = fit_bpcox_predict(X[tr], surv.iloc[tr], X[va], bp_params)

                _, rsf_va_s = standardize_risk_by_train(rsf_tr, rsf_va)
                _, bp_va_s = standardize_risk_by_train(bp_tr, bp_va)

                risk = w * rsf_va_s + (1 - w) * bp_va_s
                vals.append(calc_cindex(surv.iloc[va], risk))
            except Exception:
                pass

        score = float(np.mean(vals)) if vals else -np.inf

        rows.append({
            "rsf_weight": float(w),
            "bpcox_weight": float(1 - w),
            "cv_cindex": score
        })

        if score > best_score:
            best_w, best_score = float(w), score

    return best_w, best_score, pd.DataFrame(rows)


# =========================================================
# 8. 可视化
# =========================================================

def plot_km_curve(surv, risk, title, path):
    risk = np.asarray(risk).reshape(-1)
    high = risk >= np.median(risk)
    low = ~high

    plt.figure(figsize=(7, 6))
    kmf = KaplanMeierFitter()

    if low.sum() > 0:
        kmf.fit(surv.loc[low, "time"], surv.loc[low, "event"], label="Low risk")
        kmf.plot_survival_function(ci_show=True)

    if high.sum() > 0:
        kmf.fit(surv.loc[high, "time"], surv.loc[high, "event"], label="High risk")
        kmf.plot_survival_function(ci_show=True)

    pval, hr = calc_logrank_hr(surv, risk)

    plt.title(f"{title}\nlog-rank p={pval:.3e}, HR={hr:.3f}")
    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_risk_distribution(surv, risk, title, path):
    risk = np.asarray(risk).reshape(-1)
    order = np.argsort(risk)

    plt.figure(figsize=(10, 7))

    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(risk)), risk[order], color="#4C72B0")
    plt.axhline(np.median(risk), color="red", linestyle="--")
    plt.ylabel("Risk score")
    plt.title(title)

    plt.subplot(2, 1, 2)
    plt.scatter(
        np.arange(len(risk)),
        surv["time"].values[order],
        c=surv["event"].values[order],
        cmap="coolwarm",
        s=20
    )
    plt.xlabel("Patients ordered by risk")
    plt.ylabel("Survival time")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_model_comparison(df, out_dir, cancer):
    plt.figure(figsize=(13, 6))
    plt.bar(df["Model"], df["C_index"], color="#4C72B0")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("C-index")
    plt.title(f"{cancer} Model Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cancer}_model_cindex_comparison.png"))
    plt.close()

    heat_cols = ["C_index", "HR_high_vs_low"]
    temp = df.set_index("Model")[heat_cols]

    plt.figure(figsize=(8, max(4, len(temp) * 0.45)))
    if SEABORN_AVAILABLE:
        sns.heatmap(temp, annot=True, fmt=".3f", cmap="YlOrRd")
    else:
        plt.imshow(temp.values, cmap="YlOrRd")
        plt.colorbar()
        plt.xticks(range(len(heat_cols)), heat_cols)
        plt.yticks(range(len(temp)), temp.index)

    plt.title(f"{cancer} Metrics Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cancer}_metrics_heatmap.png"))
    plt.close()


def plot_gene_corr_heatmap(expr, genes, path, title):
    if len(genes) <= 1:
        return

    corr = np.nan_to_num(np.corrcoef(expr[genes].values.T))

    plt.figure(figsize=(10, 8))

    if SEABORN_AVAILABLE:
        sns.heatmap(corr, cmap="coolwarm", center=0, xticklabels=genes, yticklabels=genes)
    else:
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(genes)), genes, rotation=90)
        plt.yticks(range(len(genes)), genes)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_expression_heatmap(expr, risk, genes, path):
    if len(genes) == 0:
        return

    order = np.argsort(risk)
    mat = expr[genes].iloc[order].values
    mat = StandardScaler().fit_transform(mat)

    plt.figure(figsize=(12, 8))

    if SEABORN_AVAILABLE:
        sns.heatmap(mat.T, cmap="vlag", center=0, xticklabels=False, yticklabels=genes)
    else:
        plt.imshow(mat.T, aspect="auto", cmap="coolwarm")
        plt.colorbar()
        plt.yticks(range(len(genes)), genes)

    plt.xlabel("Patients ordered by risk")
    plt.ylabel("Selected genes")
    plt.title("Selected gene expression heatmap")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_risk_correlation(risk_named, out_dir, cancer):
    if len(risk_named) < 2:
        return

    names = list(risk_named.keys())
    mat = np.vstack([risk_named[n] for n in names])
    corr = np.nan_to_num(np.corrcoef(mat))

    df = pd.DataFrame(corr, index=names, columns=names)

    safe_to_csv(
        df,
        os.path.join(out_dir, f"{cancer}_risk_score_correlation.csv"),
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(12, 10))

    if SEABORN_AVAILABLE:
        sns.heatmap(df, cmap="RdBu_r", center=0)
    else:
        plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(names)), names, rotation=90)
        plt.yticks(range(len(names)), names)

    plt.title(f"{cancer} Risk Score Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cancer}_risk_score_correlation_heatmap.png"))
    plt.close()


# =========================================================
# 9. 主流程
# =========================================================

def analyze_selected_genes(expr_train, genes, out_dir, tag):
    if len(genes) > 1:
        corr = np.nan_to_num(np.corrcoef(expr_train[genes].values.T))
        upper = np.abs(corr)[np.triu_indices_from(corr, k=1)]
        mean_red = float(np.mean(upper)) if len(upper) else 0.0
        max_red = float(np.max(upper)) if len(upper) else 0.0
    else:
        mean_red, max_red = 0.0, 0.0

    summary = {
        "tag": tag,
        "gene_num": int(len(genes)),
        "mean_abs_pairwise_corr": mean_red,
        "max_abs_pairwise_corr": max_red
    }

    save_json(summary, os.path.join(out_dir, f"{tag}_selected_gene_analysis_summary.json"))

    return summary


def run_one_cancer(cancer):
    out_dir = os.path.join(OUTPUT_ROOT, cancer)
    ensure_dir(out_dir)

    expr, surv, info = load_cancer_data(cancer)
    save_json(info, os.path.join(out_dir, f"{cancer}_data_info.json"))

    stratify = surv["event"] if surv["event"].value_counts().min() >= 2 else None

    expr_train, expr_test, surv_train, surv_test = train_test_split(
        expr,
        surv,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify
    )

    print("train:", expr_train.shape, "test:", expr_test.shape)

    score_df = univariate_cox_scores(expr_train, surv_train, MAX_UNIVARIATE_GENES)

    safe_to_csv(
        score_df,
        os.path.join(out_dir, f"{cancer}_train_univariate_cox_scores.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    top_genes = score_df.head(SELECTED_GENE_NUM)["gene"].tolist()

    random_pool = score_df.head(min(MAX_QUBO_CANDIDATE_GENES, len(score_df)))["gene"].tolist()
    random_genes = random.sample(random_pool, min(SELECTED_GENE_NUM, len(random_pool)))

    qubo_nored_genes, _ = select_genes_by_qubo(
        expr_train,
        score_df,
        out_dir,
        tag="qubo_no_redundancy",
        use_redundancy=False,
        redundancy_weight=0.0
    )

    qubo_full_genes, _ = select_genes_by_qubo(
        expr_train,
        score_df,
        out_dir,
        tag="qubo_full",
        use_redundancy=True,
        redundancy_weight=QUBO_REDUNDANCY_WEIGHT
    )

    feature_sets = {
        "RandomGenes": random_genes,
        "TopCoxGenes": top_genes,
        "QUBO_NoRedundancy": qubo_nored_genes,
        "QUBO_Full": qubo_full_genes
    }

    analysis_rows = []

    for tag, genes in feature_sets.items():
        analysis_rows.append(analyze_selected_genes(expr_train, genes, out_dir, tag))

        plot_gene_corr_heatmap(
            expr_train,
            genes,
            os.path.join(out_dir, f"{tag}_gene_correlation_heatmap.png"),
            f"{cancer} {tag}"
        )

    safe_to_csv(
        pd.DataFrame(analysis_rows),
        os.path.join(out_dir, f"{cancer}_selected_gene_analysis_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    results = []
    risk_named = {}
    fusion_rows = []

    for tag, genes in feature_sets.items():
        print_section(f"{cancer} 特征方案: {tag}")

        X_train_raw = expr_train[genes].values
        X_test_raw = expr_test[genes].values

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw))
        X_test = scaler.transform(imputer.transform(X_test_raw))

        # CoxPH
        try:
            best = cv_tune_cox(X_train, surv_train, 0.0)
            _, _, risk = fit_cox_predict(X_train, surv_train, X_test, best["penalizer"], 0.0)

            name = f"{tag}-CoxPH"
            res = evaluate_survival_model(name, surv_test, risk)
            res.update({
                "FeatureSet": tag,
                "FeatureNum": len(genes),
                "CV_C_index": best["cv_cindex"],
                "FusionWeight_RSF": np.nan,
                "FusionWeight_BPCox": np.nan
            })

            results.append(res)
            risk_named[name] = risk
        except Exception:
            traceback.print_exc()

        # Lasso-Cox
        try:
            best = cv_tune_cox(X_train, surv_train, 1.0)
            _, _, risk = fit_cox_predict(X_train, surv_train, X_test, best["penalizer"], 1.0)

            name = f"{tag}-LassoCox"
            res = evaluate_survival_model(name, surv_test, risk)
            res.update({
                "FeatureSet": tag,
                "FeatureNum": len(genes),
                "CV_C_index": best["cv_cindex"],
                "FusionWeight_RSF": np.nan,
                "FusionWeight_BPCox": np.nan
            })

            results.append(res)
            risk_named[name] = risk
        except Exception:
            traceback.print_exc()

        # PCA-Cox
        try:
            n_comp = min(PCA_COMPONENTS, X_train.shape[1], X_train.shape[0] - 2)
            if n_comp >= 1:
                pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
                Ztr = pca.fit_transform(X_train)
                Zte = pca.transform(X_test)

                best = cv_tune_cox(Ztr, surv_train, 0.0)
                _, _, risk = fit_cox_predict(Ztr, surv_train, Zte, best["penalizer"], 0.0)

                name = f"{tag}-PCA-Cox"
                res = evaluate_survival_model(name, surv_test, risk)
                res.update({
                    "FeatureSet": tag,
                    "FeatureNum": n_comp,
                    "CV_C_index": best["cv_cindex"],
                    "FusionWeight_RSF": np.nan,
                    "FusionWeight_BPCox": np.nan
                })

                results.append(res)
                risk_named[name] = risk
        except Exception:
            traceback.print_exc()

        # RF / RSF
        best_rsf = None
        try:
            best_rsf = cv_tune_rsf(X_train, surv_train)
            _, rsf_train, rsf_test = fit_rsf_predict(X_train, surv_train, X_test, best_rsf)

            name = f"{tag}-RSF"
            res = evaluate_survival_model(name, surv_test, rsf_test)
            res.update({
                "FeatureSet": tag,
                "FeatureNum": len(genes),
                "CV_C_index": best_rsf["cv_cindex"],
                "FusionWeight_RSF": np.nan,
                "FusionWeight_BPCox": np.nan
            })

            results.append(res)
            risk_named[name] = rsf_test
        except Exception:
            traceback.print_exc()
            rsf_train, rsf_test = None, None

        # BP-Cox
        best_bp = None
        try:
            best_bp = cv_tune_bpcox(X_train, surv_train)
            _, bp_train, bp_test = fit_bpcox_predict(X_train, surv_train, X_test, best_bp)

            name = f"{tag}-BP-Cox"
            res = evaluate_survival_model(name, surv_test, bp_test)
            res.update({
                "FeatureSet": tag,
                "FeatureNum": len(genes),
                "CV_C_index": best_bp["cv_cindex"],
                "FusionWeight_RSF": np.nan,
                "FusionWeight_BPCox": np.nan
            })

            results.append(res)
            risk_named[name] = bp_test
        except Exception:
            traceback.print_exc()
            bp_train, bp_test = None, None

        # Fusion
        try:
            if best_rsf is not None and best_bp is not None:
                w, w_score, wdf = tune_fusion_weight_cv(X_train, surv_train, best_rsf, best_bp)

                safe_to_csv(
                    wdf,
                    os.path.join(out_dir, f"{tag}_fusion_weight_cv.csv"),
                    index=False,
                    encoding="utf-8-sig"
                )

                _, rsf_train, rsf_test = fit_rsf_predict(X_train, surv_train, X_test, best_rsf)
                _, bp_train, bp_test = fit_bpcox_predict(X_train, surv_train, X_test, best_bp)

                _, rsf_test_s = standardize_risk_by_train(rsf_train, rsf_test)
                _, bp_test_s = standardize_risk_by_train(bp_train, bp_test)

                fusion = w * rsf_test_s + (1 - w) * bp_test_s

                name = f"{tag}-RSF-BPCox-Fusion"
                res = evaluate_survival_model(name, surv_test, fusion)
                res.update({
                    "FeatureSet": tag,
                    "FeatureNum": len(genes),
                    "CV_C_index": w_score,
                    "FusionWeight_RSF": w,
                    "FusionWeight_BPCox": 1 - w
                })

                results.append(res)
                risk_named[name] = fusion

                fusion_rows.append({
                    "FeatureSet": tag,
                    "FusionWeight_RSF": w,
                    "FusionWeight_BPCox": 1 - w,
                    "CV_C_index": w_score
                })
        except Exception:
            traceback.print_exc()

    result_df = pd.DataFrame(results)

    if result_df.empty:
        raise RuntimeError(f"{cancer} 没有成功训练任何模型")

    result_df = result_df.sort_values("C_index", ascending=False).reset_index(drop=True)

    safe_to_csv(
        result_df,
        os.path.join(out_dir, f"{cancer}_model_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    if fusion_rows:
        safe_to_csv(
            pd.DataFrame(fusion_rows),
            os.path.join(out_dir, f"{cancer}_fusion_weights.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    print(result_df)

    plot_model_comparison(result_df, out_dir, cancer)
    plot_risk_correlation(risk_named, out_dir, cancer)

    main_name = "QUBO_Full-RSF-BPCox-Fusion"

    if main_name in risk_named:
        risk = risk_named[main_name]

        plot_km_curve(
            surv_test,
            risk,
            f"{cancer} {main_name}",
            os.path.join(out_dir, f"{cancer}_{safe_name(main_name)}_KM.png")
        )

        plot_risk_distribution(
            surv_test,
            risk,
            f"{cancer} {main_name} Risk Distribution",
            os.path.join(out_dir, f"{cancer}_{safe_name(main_name)}_risk_distribution.png")
        )

        plot_expression_heatmap(
            expr_test,
            risk,
            qubo_full_genes,
            os.path.join(out_dir, f"{cancer}_{safe_name(main_name)}_expression_heatmap.png")
        )

    km_dir = os.path.join(out_dir, "KM_curves")
    ensure_dir(km_dir)

    for name, risk in risk_named.items():
        try:
            plot_km_curve(
                surv_test,
                risk,
                f"{cancer} {name}",
                os.path.join(km_dir, f"{safe_name(name)}_KM.png")
            )
        except Exception:
            pass

    summary = {
        "cancer": cancer,
        "best_model": result_df.iloc[0].to_dict(),
        "main_model": main_name,
        "main_model_result": result_df[result_df["Model"] == main_name].iloc[0].to_dict()
        if main_name in result_df["Model"].values else None,
        "selected_genes_qubo_full": qubo_full_genes,
        "selected_genes_qubo_no_redundancy": qubo_nored_genes,
        "target_endpoint": TARGET_ENDPOINT,
        "allow_mixed_endpoints": ALLOW_MIXED_ENDPOINTS,
        "use_cohort_zscore": USE_COHORT_ZSCORE,
        "end_time": now_str()
    }

    save_json(summary, os.path.join(out_dir, f"{cancer}_run_summary.json"))

    return result_df, summary


def plot_global_results(global_df):
    if global_df.empty:
        return

    safe_to_csv(
        global_df,
        os.path.join(OUTPUT_ROOT, "global_model_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(16, 7))
    labels = global_df["Cancer"] + "-" + global_df["Model"]
    plt.bar(labels, global_df["C_index"], color="#4C72B0")
    plt.xticks(rotation=70, ha="right", fontsize=7)
    plt.ylabel("C-index")
    plt.title("Global Model C-index Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, "global_model_cindex.png"))
    plt.close()

    pivot = global_df.pivot_table(
        index="Cancer",
        columns="Model",
        values="C_index",
        aggfunc="mean"
    )

    plt.figure(figsize=(18, max(5, pivot.shape[0] * 0.8)))

    if SEABORN_AVAILABLE:
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1)
    else:
        plt.imshow(pivot.values, cmap="YlOrRd", vmin=0, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
        plt.yticks(range(len(pivot.index)), pivot.index)

    plt.title("Global C-index Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, "global_cindex_heatmap.png"))
    plt.close()


def main():
    args = parse_args()
    apply_config(args)

    print_section("方向二：QBM-inspired QUBO + RF/RSF + BP-Cox 融合预后模型开始")

    print("DATA_ROOT:", DATA_ROOT)
    print("OUTPUT_ROOT:", OUTPUT_ROOT)
    print("CANCERS:", CANCERS)
    print("DEVICE:", DEVICE)
    print("SKSURV_AVAILABLE:", SKSURV_AVAILABLE)
    print("KAIWU_AVAILABLE:", KAIWU_AVAILABLE)
    print("TARGET_ENDPOINT:", TARGET_ENDPOINT)
    print("ALLOW_MIXED_ENDPOINTS:", ALLOW_MIXED_ENDPOINTS)
    print("USE_COHORT_ZSCORE:", USE_COHORT_ZSCORE)

    save_json(
        {
            "data_root": DATA_ROOT,
            "output_root": OUTPUT_ROOT,
            "cancers": CANCERS,
            "method": "Direction2: QBM-inspired QUBO + RF/RSF + BP-Cox Survival Fusion",
            "vae_used": False,
            "random_seed": RANDOM_SEED,
            "cv_folds": CV_FOLDS,
            "test_size": TEST_SIZE,
            "selected_gene_num": SELECTED_GENE_NUM,
            "max_qubo_candidate_genes": MAX_QUBO_CANDIDATE_GENES,
            "target_endpoint": TARGET_ENDPOINT,
            "allow_mixed_endpoints": ALLOW_MIXED_ENDPOINTS,
            "use_cohort_zscore": USE_COHORT_ZSCORE,
            "endpoint_fallback_if_too_small": ENDPOINT_FALLBACK_IF_TOO_SMALL,
            "kaiwu_available": KAIWU_AVAILABLE,
            "sksurv_available": SKSURV_AVAILABLE,
            "force_rf_fallback": FORCE_RF_FALLBACK,
            "device": DEVICE
        },
        os.path.join(OUTPUT_ROOT, "experiment_config.json")
    )

    all_results = []
    summaries = []

    for cancer in CANCERS:
        try:
            df, summary = run_one_cancer(cancer)
            df.insert(0, "Cancer", cancer)
            all_results.append(df)
            summaries.append(summary)

        except Exception as e:
            traceback.print_exc()
            summaries.append({
                "cancer": cancer,
                "error": str(e),
                "time": now_str()
            })

    if all_results:
        global_df = pd.concat(all_results, axis=0, ignore_index=True)
        plot_global_results(global_df)

    save_json(summaries, os.path.join(OUTPUT_ROOT, "global_run_summaries.json"))

    print_section("全部完成")
    print("输出目录:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
