# -*- coding: utf-8 -*-
"""
QUBO-Fusion 统一参数文件
============================================================
所有脚本的默认参数集中在本文件调整。运行命令为：
    python main.py --mode=<mode> [额外命令行参数]

命令行参数可覆盖本文件中的默认值（每个脚本均支持 argparse）。
============================================================
"""

import os

# numpy 可选：核心实验必须安装；仅运行数据下载（download_data.py）时允许缺失
try:
    import numpy as np
except ImportError:
    np = None

# ============================================================
# 1. 路径配置
# ============================================================
# 原始数据根目录（含各癌种子目录的 mrna.rda）
DATA_ROOT = "data"

# 实验结果根目录
RESULT_ROOT = "result"

# 各步骤输出目录（core_qubo_fusion.py 的输出是后续步骤的输入）
OUTPUT_MAIN = os.path.join(RESULT_ROOT, "main")            # 核心主实验
OUTPUT_BASELINES = os.path.join(RESULT_ROOT, "baselines")  # 强基线补充
OUTPUT_EXTENSIONS = os.path.join(RESULT_ROOT, "extensions")  # 扩展验证
OUTPUT_BRCA = os.path.join(RESULT_ROOT, "brca")            # BRCA 专项验证

# ============================================================
# 2. 数据
# ============================================================
# SurvivalML 数据集（Synapse）
SYNAPSE_ID = "syn58922557"

# 参与实验的癌种
CANCERS = ["BRCA", "CRC", "LUAD", "GBM", "LUSC", "DLBC"]

# ============================================================
# 3. 实验设置
# ============================================================
# 主实验随机种子（BRCA 专项验证使用 12 个随机种子，见第 5 节）
RANDOM_SEED = 42

# 同分布 7:3 随机划分与交叉验证
TEST_SIZE = 0.3
CV_FOLDS = 3

# 样本 / 基因 / 生存时间过滤
MIN_SAMPLE_NUM = 10
MIN_GENE_NUM = 10
MIN_SURVIVAL_TIME = 1e-8

# 生存终点
ALLOW_MIXED_ENDPOINTS = False
TARGET_ENDPOINT = "OS"
MIN_COHORT_EVENT_NUM = 3
USE_COHORT_ZSCORE = True
ENDPOINT_FALLBACK_IF_TOO_SMALL = False
MIN_TOTAL_SAMPLE_AFTER_ENDPOINT_FILTER = 80

# 基因选择
MAX_UNIVARIATE_GENES = 2000       # 单变量 Cox 预筛上限
MAX_MERGED_GENES = 10000          # 合并队列基因数上限
MAX_QUBO_CANDIDATE_GENES = 80     # QUBO 候选基因数
SELECTED_GENE_NUM = 20            # QUBO 最终选择的基因数

# QUBO 目标权重：min  w_rel*相关性项 + w_red*冗余项 + w_card*基数项
QUBO_RELEVANCE_WEIGHT = 1.0
QUBO_REDUNDANCY_WEIGHT = 0.35
QUBO_CARDINALITY_WEIGHT = 5.0

# 模拟退火求解器
USE_KAIWU_FIRST = True            # 优先尝试 Kaiwu SDK，失败回退本地模拟退火
SA_STEPS = 4000                   # 退火迭代步数
SA_T0 = 5.0                       # 初始温度
SA_T1 = 0.01                      # 终止温度

# PCA-Cox 降维
PCA_COMPONENTS = 8

# RSF 超参数网格（交叉验证选择）
RSF_PARAM_GRID = [
    {"n_estimators": 80, "max_depth": 8, "min_samples_leaf": 5},
    {"n_estimators": 120, "max_depth": 10, "min_samples_leaf": 5},
    {"n_estimators": 150, "max_depth": None, "min_samples_leaf": 8},
]
RSF_N_JOBS = 1
RSF_PREDICT_CHUNK_SIZE = 128
# 稳定起见默认使用 RF fallback，不强依赖 sksurv。
# 若已安装 scikit-survival 并想使用 RandomSurvivalForest，请改为 False。
FORCE_RF_FALLBACK = True

# BP-Cox 超参数网格（交叉验证选择）
BPCOX_PARAM_GRID = [
    {"hidden_dim": 32, "dropout": 0.20, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 100},
    {"hidden_dim": 64, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 120},
]

# RSF 与 BP-Cox 融合权重搜索网格（CoxPH 风险分数的线性加权）
if np is not None:
    FUSION_WEIGHT_GRID = np.linspace(0, 1, 21).tolist()
else:
    FUSION_WEIGHT_GRID = [i / 20.0 for i in range(21)]

# ============================================================
# 4. 扩展验证（three_extension_analyses.py）
# ============================================================
# QUBO vs FullModel
FULL_MODEL_GENES = 1000           # FullModel 特征数（500~2000）
HIGH_DIM_COX_LIMIT = 500          # 高维 Cox 的最大特征数

# 同癌种跨数据集：最小样本 / 事件 / 基因门槛
MIN_CROSS_TRAIN_SAMPLE = 100
MIN_CROSS_TEST_SAMPLE = 100
MIN_CROSS_EVENT = 20
MIN_CROSS_GENE = 5

# 跨癌种迁移：最小样本 / 事件 / 基因门槛
MIN_TRANSFER_TRAIN_SAMPLE = 50
MIN_TRANSFER_TEST_SAMPLE = 50
MIN_TRANSFER_EVENT = 5
MIN_TRANSFER_GENE = 5

# ============================================================
# 5. BRCA 专项验证（brca_validation.py）
# ============================================================
BRCA_SEEDS = [0, 7, 13, 42, 123, 256, 512, 1024, 2024, 9999, 12345, 77777]
BRCA_N_SEEDS = 12
