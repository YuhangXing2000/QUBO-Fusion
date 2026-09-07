# 【QUBO-Fusion】课题

## 课题简介

预后生物标志物的发现对肿瘤精准治疗至关重要，然而现有方法在多队列验证中的可重复性不足。本文提出 **QUBO-Fusion** 框架：将**二次无约束二值优化（Quadratic Unconstrained Binary Optimization, QUBO）**应用于基因选择，并结合**随机生存森林（RSF）**与**反向传播-Cox 回归（BP-Cox）**的加权融合策略，构建跨癌种生存风险预测模型。

本课题在 [SurvivalML](https://github.com/Zaoqu-Liu/SurvivalML) 平台的 6 个癌种数据集（BRCA、CRC、LUAD、GBM、LUSC、DLBC，共 14,781 例样本）上复现并验证该框架，覆盖同癌种跨数据集验证、同分布上界实验、跨癌种迁移验证、BRCA 专项强基线验证与预后基因癌种特异性分析等系统性实验。

> 说明：本研究使用**经典模拟退火算法**求解 QUBO 问题（可选接入 Kaiwu SDK / 量子退火求解器），不依赖量子硬件，重点在于 QUBO 组合优化框架本身的方法学价值。

### 核心发现

| 发现 | 结果 |
|---|---|
| 特征精简 | QUBO 仅用约 **20 个基因**（~4%）即可达到约 500 基因 FullModel 的主体性能 |
| 冗余控制 | QUBO 所选基因内部冗余性较随机选择平均降低约 **50%**（45%~60%） |
| 同分布上界 | 24 种基因选择-预测模型组合中，含 QUBO 的组合在 **5/6** 癌种取得最优 C-index；BRCA 最突出（QUBO_NoRed-Fusion，C-index 0.682，p=3.12×10⁻⁸，HR=2.412） |
| 跨数据集稳健性 | LUAD（中位 C-index≈0.60）与 BRCA（≈0.58）预后信号跨批次可迁移；DLBC、LUSC 稳健性有限（≈0.50） |
| 癌种特异性 | 预后基因高度癌种特异（Jaccard 相似度均值 ≈ 0.009），支持癌种定制化策略 |
| BRCA 专项验证 | 12 随机种子 × 9 特征选择 × 6 模型 = 648 个评估；QUBO-Full+CoxPH 以 0.631 居全局第一，相对 TopCox 提升 +0.040（配对 t 检验 p=0.018） |

完整方法、结果与讨论见 [docs/](docs/README.md)。

---

## 项目结构

```
QUBO-Fusion/                                  # 项目目录
├── README.md                                 # 项目说明文档
├── requirements.txt                          # 依赖包列表
├── params.py                                 # 统一参数文件（路径 / 癌种 / 超参 / QUBO 权重）
├── main.py                                   # 主程序入口（--mode=...）
├── download_data.py                          # 数据下载代码（SurvivalML / Synapse）
├── core_qubo_fusion.py                       # 核心方法：QUBO 基因选择 + RSF/BP-Cox/Fusion + 同分布主实验 + 生物学分析
├── strong_baselines.py                       # 强基线补充代码（mRMR / RSF_Importance）
├── three_extension_analyses.py               # 扩展验证代码（跨癌种迁移 / 同癌种跨数据集 / QUBO vs FullModel）
├── brca_validation.py                        # BRCA 专项验证代码（12 随机种子强基线）
├── pan_cancer_summary.py                     # 泛癌种汇总代码（基因频率 / Jaccard 分析）
├── enhanced_visualization.py                 # 增强可视化代码（出版级图表精修）
├── data/                                     # 原始数据（由 download_data.py 生成）
├── result/                                   # 实验结果（按步骤分子目录）
└── docs/                                     # 研究文档（主论文 + 3 份补充材料 + 代码报告 + 全部图）
```

---

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 可选：RSF 与 QUBO 求解器（缺失时自动回退随机森林回归 / 本地模拟退火）
pip install scikit-survival kaiwu
```

### 2. 运行示例

在 `params.py` 中调整路径参数与实验参数（数据目录、癌种列表、QUBO 权重、超参网格等）。

**数据准备**（需要 Synapse 账号与个人访问令牌）：

```bash
python main.py --mode=data --token <YOUR_TOKEN>
```

**核心主实验**（先运行，产出主结果）：

```bash
python main.py --mode=main
```

运行后可生成各癌种的模型结果表（`result/main/<CANCER>/<cancer>_model_results.csv`）、QUBO 基因列表、KM 曲线、热图等，对应论文表 3-6 与图 4-20。

**后续分析**（依赖核心主实验输出）：

```bash
# 强基线补充（mRMR / RSF_Importance）
python main.py --mode=baselines

# 扩展验证（跨癌种迁移 / 同癌种跨数据集 / QUBO vs FullModel）
python main.py --mode=extensions

# BRCA 专项强基线验证（12 随机种子，耗时较长）
python main.py --mode=brca

# 泛癌种汇总与基因 Jaccard 分析
python main.py --mode=pan

# 可选：出版级可视化精修
python main.py --mode=visualize

# 一键运行 main → baselines → extensions → pan → visualize
python main.py --mode=all
```

**临时覆盖参数**：绝大多数参数在 `params.py` 中统一调整；需要临时覆盖时，在 `--mode` 后追加对应脚本支持的命令行参数即可，例如：

```bash
python main.py --mode=brca --n_seeds 3
python main.py --mode=main --cancers BRCA,LUAD
python main.py --mode=main --output result/main_test
```

各脚本详细说明、输出文件与论文表图对照见 [docs/code_usage_report.md](docs/code_usage_report.md)。

---

## 算法说明

### 1. QUBO 基因选择模型

#### 1.1 QUBO 问题的一般形式

QUBO 是量子退火计算的标准问题形式，一个 QUBO 问题可以表述为最小化如下二次目标函数（二值变量 `x_i ∈ {0,1}`，`x_i = 1` 表示选择第 `i` 个基因）：

<img src="docs/formulas/f1.png" alt="公式 f1">

其中 `Q_ii` 为对角项（一元偏好系数），`Q_ij` 为交叉项（二元交互系数）。QUBO 虽是量子退火的标准形式，本研究全程使用经典模拟退火求解器，不涉及量子硬件实现。

#### 1.2 单变量 Cox 综合评分

对候选基因进行单变量 Cox 回归打分。与传统的仅使用 p 值不同，采用综合评分策略，同时考虑三个维度——Cox 回归的 `|z|` 统计量、单基因 C-index 和基因表达方差：

<img src="docs/formulas/f2.png" alt="公式 f2">

其中 `rank(·)` 表示基因 `i` 在候选基因集合中的百分位排名，`z_i` 为单变量 Cox 回归的 z 统计量，`c_i` 为单基因 C-index，`v_i` 为基因表达方差（综合函数的具体形式见论文方法（一））。该策略中：`|z|` 统计量反映基因与生存的统计关联强度，C-index 衡量基因判别能力，方差确保所选基因具有足够的表达变异度。

#### 1.3 QUBO 矩阵构造

基于综合评分 `s_i` 与基因间 Pearson 相关系数 `ρ(g_i, g_j)`，构造 QUBO 矩阵的对角项与交叉项：

<img src="docs/formulas/f3a.png" alt="公式 f3a">

<img src="docs/formulas/f3b.png" alt="公式 f3b">

<img src="docs/formulas/f3c.png" alt="公式 f3c">

基数约束项以权重 `w_card = 5.0` 加入目标函数，使选中基因数接近目标值 `k`。

对角项的负号确保高评分基因被优先选中；交叉项的正相关系数惩罚冗余选择；基数约束项使选中基因数接近目标值 `k`。超参数默认 `w_rel = 1.0`，`w_red = 0.35`，`w_card = 5.0`，`k = 20`（可在 `params.py` 调整）。

#### 1.4 两阶段候选集与模拟退火求解

直接在全基因组（`n > 10,000`）上构建 QUBO 矩阵计算量过大，采用两阶段策略：先按综合评分排序保留前 80 个候选基因（`MAX_QUBO_CANDIDATE_GENES`），再在该子集上构建 `80 × 80` 的 QUBO 矩阵。求解采用经典模拟退火：

<img src="docs/formulas/f4.png" alt="公式 f4">

即温度从 `T_0 = 5.0` 指数退火至 `T_1 = 0.01`，共 `N = 4000` 步；若安装了 Kaiwu SDK（`USE_KAIWU_FIRST=True`），可优先接入量子退火求解器。

#### 1.5 两种 QUBO 变体

| 变体 | 冗余惩罚 | 目标 |
|---|---|---|
| **QUBO_Full** | `w_red = 0.35`（含） | 选出 20 个兼顾预测能力与信息多样性的基因 |
| **QUBO_NoRedundancy** | `w_red = 0.0`（不含） | 选出 20 个以预后信号强度为主要标准的基因 |

### 2. RSF 随机生存森林

随机生存森林（Random Survival Forest, RSF）是随机森林在生存分析领域的扩展：每棵树基于自助采样（bootstrap）构建，并在每个节点分裂时随机选择候选特征子集，最终预测为所有树的集成累积风险函数：

<img src="docs/formulas/f5.png" alt="公式 f5">

其中 `B` 为树的数量，`H_b(t|x)` 为第 `b` 棵生存树的累积风险函数（Nelson–Aalen 估计）。RSF 无需假设比例风险条件，能自动建模非线性效应与高阶交互。超参通过 3 折 CV 从 `RSF_PARAM_GRID` 选择（树数 80/120/150，最大深度 8/10/None，最小叶节点样本 5/8）；sksurv 缺失时自动回退随机森林回归 + 生存风险映射（`FORCE_RF_FALLBACK=True`）。

### 3. BP-Cox 深度学习 Cox 模型

BP-Cox 将反向传播神经网络与 Cox 比例风险框架相结合：网络编码器 `φ(x; θ)`（输入 → hidden_dim → hidden_dim/2 → 1，LayerNorm + ReLU + Dropout，p = 0.20~0.25）提取高阶非线性特征，Cox 层输出风险对数 `η_i = φ(x_i; θ)ᵀ β`。训练采用 Cox **负对数偏似然损失**：

<img src="docs/formulas/f6.png" alt="公式 f6">

其中 `D` 为事件集合，`R(t_i)` 为 `t_i` 时刻的风险集。优化采用 Adam（初始学习率 `1×10⁻³`，权重衰减 `1×10⁻⁴`），训练 100~120 个 epoch，早停 patience=25，取训练过程中损失最低的 checkpoint。

### 4. RSF-BP-Cox 加权融合

RSF 擅长捕获特征间非线性交互，BP-Cox 擅长提取高阶抽象特征。为利用两者的互补优势，先对两模型的风险评分做 Z-score 标准化，再线性加权：

<img src="docs/formulas/f7a.png" alt="公式 f7a">

<img src="docs/formulas/f7b.png" alt="公式 f7b">

其中 `μ`、`σ` 在训练集上估计，确保两个模型的评分在同一量纲上融合。

**融合权重选择算法**：权重 `w` 通过 3 折交叉验证在验证集上确定——在 21 点网格 `W = {0, 0.05, 0.10, …, 1.0}` 上搜索，以验证集 C-index 均值为选择标准：

<img src="docs/formulas/f8.png" alt="公式 f8">

其中 `K` 为 CV 折数，`C_k(w)` 为第 `k` 折验证集上权重 `w` 对应的 C-index。最终融合模型在完整训练集上重新训练，并在独立测试集上评估。

### 5. 评估指标

#### 5.1 一致性指数（C-index）

<img src="docs/formulas/f9.png" alt="公式 f9">

其中 `T_i` 为观察时间，`δ_i` 为事件指示符（1=事件，0=删失），`r̂` 为预测风险评分。C-index ∈ [0,1]，0.5 表示随机预测，1 表示完美预测。

#### 5.2 基因冗余度

QUBO 所选基因集合 `S` 的内部冗余度定义为基因两两 Pearson 相关系数绝对值的平均：

<img src="docs/formulas/f10.png" alt="公式 f10">

实验中 QUBO_Full 选出的基因集合在全部 6 个癌种上均具有更低的平均相关系数，较随机选择平均降低约 50%（范围 45%~60%）。

#### 5.3 Jaccard 相似度（癌种特异性）

<img src="docs/formulas/f11.png" alt="公式 f11">

用于量化不同癌种（或不同数据集）间 QUBO 选择基因集合的重叠程度。实验测得跨癌种 Jaccard 均值 ≈ 0.009（绝大多数癌种对为 0.000，仅 3 对为 0.026），证实预后基因高度癌种特异。

#### 5.4 HR 与 log-rank 检验

按预测风险中位数将样本分为高/低风险组，风险比（Hazard Ratio, HR）由 Cox 模型估计，组间生存差异用 log-rank 检验的 p 值评估（KM 曲线可视化）。

### 6. 同分布上界与参数选择

同分布实验（7:3 随机划分）用于评估方法潜力上界；跨数据集验证用于评估真实场景可迁移性。两套实验的 C-index 绝对值口径不同（CV 折数、FullModel 上限、权重网格点数不同），需按论文口径分开解读，详见 [docs/main_paper.md](docs/main_paper.md)。

---

## 数据与复现说明

- **数据来源**：SurvivalML 平台（Synapse 数据集 ID `syn58922557`），6 癌种转录组 + 临床生存数据，生存终点统一为 OS。
- **主实验随机种子**：42；BRCA 专项验证另用 12 个随机种子（0, 7, 13, 42, 123, 256, 512, 1024, 2024, 9999, 12345, 77777）。
- **两套参数口径不可直接比较**：主实验（CV=3、Fusion 权重 21 点、RSF 三组参数）与补充验证快速配置（CV=2、FullModel 上限 500、权重 6 点）的 C-index 绝对值需按论文口径分开解读。
- **原始下载脚本缺失**：原始压缩包未包含数据下载脚本，`download_data.py` 依据代码使用报告的描述重建，未经真实 Synapse 账号端到端测试，使用前请核对 Synapse ID 与目录布局。
- **未纳入仓库的脚本**：`癌症4.py`、`癌症qbm代码.py`（QBM-VAE 方法线）、`癌症2_修改后.py`（GBK 编码旧版）、`癌症2_最终修改版.py`（调度版）、`癌症3_修改后.py`（增强版，功能已由 `three_extension_analyses.py` 与 `pan_cancer_summary.py` 覆盖）。

---

## 引用

如果你使用了本仓库，请引用：

```
QUBO-Fusion: QUBO-based gene selection with RSF-BP-Cox fusion
for multi-cancer prognostic biomarker discovery.
```

数据与基准参考：

- SurvivalML: A comprehensive platform for prognostic biomarker discovery. https://github.com/Zaoqu-Liu/SurvivalML
- Liu Z, Deng J, Xu H, et al. Efficient discovery of robust prognostic biomarkers and signatures in solid tumors. *Cancer Letters*. 2025.

---

## 作者信息

- 作者姓名：（待填写）
- 联系方式：（待填写）

## License

[MIT](LICENSE)
