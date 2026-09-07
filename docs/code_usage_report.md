> 本文档是 QUBO-Fusion 多癌种预后研究合集的一部分。完整合集见仓库 docs/ 目录。

# 附录　代码使用报告

QUBO-Fusion 多癌种预后研究代码使用报告

——支撑论文《量子癌症.docx》全部结果所需的代码清单与使用说明——

> **整理说明（GitHub 发布版）**：本报告中的脚本名与路径已按仓库实际文件更新。原中文脚本名与发布版文件名的对应关系如下：
>
> | 原文件名 | 发布版文件名 |
> |---|---|
> | 癌症.py | download_data.py（原始文件缺失，按本报告描述重建） |
> | 癌症2.py | core_qubo_fusion.py |
> | 癌症补充脚本.py | strong_baselines.py |
> | 癌症分析.py | three_extension_analyses.py |
> | 癌症BRCA.py | brca_validation.py |
> | 癌症3.py | pan_cancer_summary.py |
> | 癌症可视化.py | enhanced_visualization.py |
>
> 原 Windows 绝对路径（D:\PPT\癌症 等）已统一改为命令行参数配置的默认相对路径（data/ 与 result/）；代码已按量子开发实验室项目规范迁移至项目根目录，并新增统一参数文件 params.py 与主程序入口 main.py（python main.py --mode=...）。


### 一、报告目的与研究背景

#### （一）研究目标

论文《量子癌症.docx》提出 QUBO-Fusion 框架：将二次无约束二值优化（Quadratic Unconstrained Binary Optimization, QUBO）用于基因选择，结合随机生存森林（RSF）与反向传播-Cox 回归（BP-Cox）的加权融合，构建跨癌种生存风险预测模型。研究基于 SurvivalML 平台的 6 个癌种数据集（BRCA、CRC、LUAD、GBM、LUSC、DLBC，共 14,781 例样本），开展了同癌种跨数据集验证（主结果）、QUBO 预筛与 FullModel 对比、同分布上界实验、跨癌种迁移验证以及预后基因癌种特异性分析等系统性实验。

#### （二）本报告内容

本报告给出支撑论文全部结果所需的最小代码集合，逐份说明每份脚本的用途、关键配置、运行方式、输出文件及其对应的论文章节，并给出推荐的运行顺序、运行环境与注意事项，供作者按流程完整复现论文结果。

### 二、代码清单（一共需要哪些代码）

经对 12 份 Python 脚本与论文逐节比对，支撑论文《量子癌症.docx》全部结果所需的代码共 6 份必需脚本、2 份可选脚本；另有 5 份脚本属于旧版、替代版或另一条方法线，论文中不需要使用。

#### （一）必需代码（6 份）

| 序号 | 脚本 | 角色 | 对应论文内容 |
|---|---|---|---|
| 1 | download_data.py | 数据下载（SurvivalML / Synapse） | 数据来源 |
| 2 | core_qubo_fusion.py | 核心方法：QUBO 基因选择 + RSF/BP-Cox/Fusion + 同分布主实验 + 生物学分析 | 方法（二）；表3/4/5/6；图4-20 |
| 3 | strong_baselines.py | 强基线补充（mRMR、RSF_Importance） | 图7/8 强基线对比 |
| 4 | three_extension_analyses.py | 三问题扩展：跨癌种迁移 / 同癌种跨数据集 / QUBO vs FullModel | 表2/3/7；图2/3/4/5/21-23 |
| 5 | brca_validation.py | BRCA 专项强基线验证 | 表4 BRCA 行；图7-9/13 |
| 6 | pan_cancer_summary.py | 跨癌种汇总与基因 Jaccard 分析 | 图19/20 |

#### （二）可选代码（2 份）

| 脚本 | 说明 |
|---|---|
| enhanced_visualization.py | 读取三问题分析产生的 CSV，生成出版级热图、网络图、柱状图等，用于论文插图精修 |
| 癌症3_修改后.py | pan_cancer_summary.py 的增强版（增加 FullModel 对照与跨癌种迁移），可替代pan_cancer_summary.py |

#### （三）不需要或可替代的代码（5 份）

| 脚本 | 情况说明 |
|---|---|
| 癌症4.py | 任务一“QBM-VAE”方法线（VAE+CoxPH），与本文档（QUBO-Fusion）是不同方法，不使用 |
| 癌症qbm代码.py | 同上，任务一“QBM-VAE”方法线，不使用 |
| 癌症2_修改后.py | core_qubo_fusion.py 的 DLBC 单癌种快速版（CV=2、权重 6 点），功能已被core_qubo_fusion.py 覆盖；且为 GBK 编码易乱码 |
| 癌症2_最终修改版.py | 三问题“断点续跑”调度版，功能与three_extension_analyses.py 重叠，非必需 |
| 癌症3_修改后.py | 未包含在本仓库中；增强版跨癌种迁移功能由 three_extension_analyses.py 与 pan_cancer_summary.py 覆盖 |

### 三、运行环境与依赖

#### （一）基础环境

| 项目 | 要求 |
|---|---|
| 操作系统 | 跨平台（Windows / Linux / macOS），路径通过命令行参数配置（默认相对路径 data/ 与 results/） |
| Python | 3.x（脚本兼容 3.8 及以上） |
| 硬件 | 建议具备 CUDA GPU 以加速 BP-Cox 训练；无 GPU 时脚本自动回退 CPU |
| 网络 | 运行download_data.py 下载数据时需要访问 Synapse 平台 |

#### （二）第三方依赖库

| 类别 | 库 | 用途 |
|---|---|---|
| 基础库 | numpy、pandas、scikit-learn | 数据处理与建模 |
| 生存分析 | lifelines | Cox 回归、log-rank 检验、C-index、KM 曲线 |
| 深度学习 | torch | BP-Cox 神经网络 |
| 可视化 | matplotlib、seaborn（可选） | 出图 |
| 数据读取 | pyreadr、rdata | 读取 SurvivalML 的 mrna.rda |
| 可选 | sksurv（scikit-survival） | RSF 随机生存森林（缺失时自动回退随机森林回归） |
| 可选 | kaiwu | QUBO 经典模拟退火求解（缺失或未授权时自动回退本地模拟退火） |
| 数据下载 | synapseclient、synapseutils | download_data.py 下载数据 |

依赖安装建议：pip install numpy pandas scikit-learn matplotlib seaborn lifelines torch pyreadr rdata sksurv synapseclient synapseutils

### 四、推荐运行流程（数据流水线）

各脚本之间存在依赖关系：core_qubo_fusion.py 是核心基础，必须最先运行；strong_baselines.py、three_extension_analyses.py、pan_cancer_summary.py 均读取core_qubo_fusion.py 的输出；brca_validation.py 独立读取原始数据；enhanced_visualization.py 依赖three_extension_analyses.py 的输出。推荐按下表顺序执行：

| 步骤 | 脚本 | 主要输入 | 主要输出 | 论文对应 |
|---|---|---|---|---|
| 1 | download_data.py | Synapse Token | 下载 mrna.rda 至 data | 数据来源 |
| 2 | core_qubo_fusion.py | data 数据 | 各癌种 model_results、QUBO 基因、图 | 表3-6；图4-20 |
| 3 | strong_baselines.py | core_qubo_fusion.py 的输出 | mRMR/RSF_Importance 强基线结果 | 图7/8 |
| 4 | three_extension_analyses.py | core_qubo_fusion.py 的输出 | 三问题结果 CSV 与图 | 表2/3/7；图2/3/4/5/21-23 |
| 5 | brca_validation.py | BRCA 原始数据 | BRCA 专项结果、图 | 表4 BRCA；图7-9/13 |
| 6 | pan_cancer_summary.py | 各癌种输出 | 泛癌汇总、基因频率、Jaccard | 图19/20 |
| 7（可选） | enhanced_visualization.py | 三问题 CSV | 出版级图 | 图2/3/21-23 等 |

说明：brca_validation.py 与第 3、4 步无严格先后依赖，可与第 2 步完成后并行运行；若追求最快复现论文主体结果，先完成第 2 步即可获得表 3/4/5/6 与图 4-20 的主要数据。

### 五、各脚本详细使用说明

#### （一）download_data.py——数据下载

1. 用途：从 Synapse 平台下载 SurvivalML 项目数据（数据集 ID：syn58922557）至本地，是全部实验的数据前置。

2. 关键配置：TOKEN 变量填入 Synapse 个人访问令牌；SYNAPSE_ID="syn58922557"；DOWNLOAD_DIR=data。

3. 运行命令：python download_data.py

4. 前置条件：需在 Synapse 注册并创建个人访问令牌；已下载过的文件不会重复下载。

5. 输出文件：data 目录下各癌种的 mrna.rda 等数据文件。

6. 依赖：synapseclient、synapseutils。

#### （二）core_qubo_fusion.py——核心主实验

1. 用途：实现 QUBO-Fusion 核心方法，包括单基因 Cox 综合评分、QUBO 基因选择（QUBO_Full 与 QUBO_NoRedundancy 两变体）、RSF/BP-Cox/Fusion 模型训练与评估、同分布上界实验（4 特征集×6 模型=24 组合）以及生物学分析（基因冗余性、KM 曲线、风险分布、表达热图）。

2. 关键配置：DATA_ROOT=data；OUTPUT_ROOT=result/main；CANCERS 为 6 癌种；QUBO 权重 w_rel=1.0、w_red=0.35、w_card=5.0；SA_STEPS=4000、SA_T0=5.0、SA_T1=0.01；TEST_SIZE=0.3；CV_FOLDS=3；FUSION_WEIGHT_GRID=np.linspace(0,1,21)。

3. 运行命令：python core_qubo_fusion.py

4. 输出文件：每癌种目录下 {cancer}_model_results.csv（含 C_index、logrank_p、HR、FusionWeight）、qubo_full_selected_genes.csv、qubo_no_redundancy_selected_genes.csv、{cancer}_train_univariate_cox_scores.csv、{cancer}_fusion_weights.csv、KM 曲线与热图；全局 global_model_results.csv。

5. 对应论文：方法（二）；表 3/4/5/6；图 4-20。

6. 注意事项：默认 FORCE_RF_FALLBACK=True（以随机森林回归替代 sksurv RSF）；Kaiwu 不可用时自动回退本地模拟退火；BP-Cox 训练依赖 torch，建议 GPU 加速。

#### （三）strong_baselines.py——强基线补充

1. 用途：为 6 个癌种补充 mRMR 与 RSF_Importance 两个强特征选择基线，训练 RSF/BP-Cox/Fusion 并与已有 QUBO 结果合并，用于论文强基线对比。

2. 关键配置：CANCER2_SCRIPT=core_qubo_fusion.py；RESULTS_ROOT=result/main（core_qubo_fusion.py 输出根目录）；OUTPUT_DIR=result/baselines。

3. 前置条件：必须先运行core_qubo_fusion.py，生成 {cancer}_model_results.csv 与 {cancer}_train_univariate_cox_scores.csv。

4. 运行命令：python strong_baselines.py

5. 输出文件：各癌种 {cancer}_model_results_with_baselines.csv；汇总图 fig1、fig2（强基线对比与 Delta）。

6. 对应论文：图 7/8 的强基线对比。

#### （四）three_extension_analyses.py——三问题扩展验证

1. 用途：独立完成三个扩展实验——问题 1 跨癌种迁移（Cancer A→B）、问题 2 同癌种跨数据集（cohort A→B）、问题 3 QUBO 预筛 vs FullModel。

2. 关键配置：CANCER2_SCRIPT=core_qubo_fusion.py；RESULTS_ROOT=result/main；OUTPUT_ROOT=result/extensions；FULL_MODEL_GENES=1000。

3. 前置条件：必须先运行core_qubo_fusion.py，生成各癌种 QUBO 基因文件。

4. 运行命令：python three_extension_analyses.py

5. 输出文件：OUTPUT_ROOT 下 01_cross_cancer_transfer、02_within_cancer_cross_dataset、03_qubo_vs_fullmodel 三个子目录的 CSV 与图，以及 analysis_report.json。

6. 对应论文：表 2/3/7；图 2/3/4/5/21-23。

7. 说明：问题 2（同癌种跨数据集）是论文主结果，产出 155 对（310 有向迁移）的 C-index 矩阵；论文表 2 的基因零重叠统计需结合该步产生的各 cohort QUBO 基因文件另行汇总。

#### （五）brca_validation.py——BRCA 专项强基线验证

1. 用途：在 BRCA 上开展 9 种特征选择（QUBO_Full、QUBO_NoRed、TopCox、Random、mRMR、RF_Importance、Lasso_Selected、ElasticNet_Selected、VarTop500）×6 种模型（CoxPH、LassoCox、PCA-Cox、RSF、BP-Cox、Fusion）的专项验证，多 seed（默认 12 个）取均值，是论文 BRCA 最优结果与强基线结论的来源。

2. 运行命令：python brca_validation.py --data_root "data" --output "result/brca" --n_seeds 12

3. 关键参数：--data_root（数据目录）、--output（输出目录）、--n_seeds（默认 12）、--skip_km（跳过 KM 图）。

4. 输出文件：results/ 下 brca_all_combo_ranking.csv、brca_fusion_paired_comparison.csv、brca_QUBO_NoRed_Fusion_vs_all_combos.csv、brca_gene_jaccard.csv、brca_redundancy_summary.csv；figures/ 下各类图；BRCA强基线验证报告.md。

5. 对应论文：表 4 的 BRCA 最优结果（QUBO_NoRed-Fusion，C-index=0.682）；图 7-9/13。

6. 注意事项：独立读取 BRCA 原始数据（不依赖core_qubo_fusion.py 输出），多 seed 训练耗时较长，建议 GPU。

#### （六）pan_cancer_summary.py——跨癌种汇总分析

1. 用途：汇总 6 个癌种的模型结果，输出各癌种最佳模型、主模型（QUBO_Full-RSF-BPCox-Fusion）跨癌种表现、消融对比、QUBO 选择基因频率与 Jaccard 相似度矩阵。

2. 关键配置：OUTPUT_ROOT=result/main（需改为core_qubo_fusion.py 的实际输出目录）；QUBO_TAGS=["qubo_full","qubo_no_redundancy"]。

3. 前置条件：必须先运行core_qubo_fusion.py。

4. 运行命令：python pan_cancer_summary.py

5. 输出文件：pan_cancer_analysis/ 下 pan_cancer_all_model_results.csv、pan_cancer_best_model_summary.csv、pan_cancer_gene_frequency_*.csv、pan_cancer_gene_jaccard_*.csv 及各类图。

6. 对应论文：图 19/20 的基因频率与 Jaccard 分析（预后基因癌种特异性，Jaccard 均值≈0.009）。

#### （七）enhanced_visualization.py——增强可视化（可选）

1. 用途：读取三问题分析产生的 CSV，生成出版级可视化图（C-index 热图网格、迁移网络图、QUBO vs FullModel 对比、稳健性箱线图等），用于论文插图精修。

2. 前置条件：必须先运行three_extension_analyses.py。

3. 运行命令：python enhanced_visualization.py

4. 输出文件：增强可视化/ 目录下各图（300 dpi）。

### 六、输出文件与论文结果对应表

下表汇总论文各表图对应的产出脚本与关键输出文件，便于按论文行文逐项核验：

| 论文内容 | 产出脚本 | 关键输出文件 |
|---|---|---|
| 表2 跨数据集基因重叠 | three_extension_analyses.py（问题2）+ 统计 | within_cancer_cross_dataset_results.csv、各 cohort QUBO 基因文件 |
| 表3 QUBO vs FullModel | three_extension_analyses.py（问题3） | qubo_vs_fullmodel_results.csv |
| 表4 各癌种最优模型 | core_qubo_fusion.py / brca_validation.py | {cancer}_model_results.csv、brca_all_combo_ranking.csv |
| 表5 基因冗余性 | core_qubo_fusion.py | {cancer}_selected_gene_analysis_summary.csv |
| 表6 融合消融 | core_qubo_fusion.py | {cancer}_model_results.csv（Fusion 行） |
| 表7 跨癌种迁移 | three_extension_analyses.py（问题1） | cross_cancer_transfer_results.csv |
| 图2/3 同癌种跨数据集 | three_extension_analyses.py（问题2） | heatmap_*.png、summary_barplot.png |
| 图4/5 QUBO vs FullModel | three_extension_analyses.py（问题3） | qubo_vs_fullmodel_*.png、delta_*.png |
| 图7/8 强基线 | brca_validation.py / strong_baselines.py | brca_*.csv、fig*.png |
| 图12 基因相关热图 | core_qubo_fusion.py | *_gene_correlation_heatmap.png |
| 图14-16 KM 曲线 | core_qubo_fusion.py | *_KM.png |
| 图17/18 风险分布/表达热图 | core_qubo_fusion.py | *_risk_distribution.png、*_expression_heatmap.png |
| 图19/20 Jaccard | pan_cancer_summary.py | pan_cancer_gene_*.csv |
| 图21-23 跨癌种迁移 | three_extension_analyses.py（问题1）/ enhanced_visualization.py | heatmap_*.png、network.png |

### 七、注意事项与常见问题

1. 数据获取：download_data.py 需要有效 Synapse Token，未下载数据前无法运行后续全部脚本。

2. 路径配置：本仓库已将全部脚本改为命令行参数配置路径（默认相对路径 data/ 与 results/），可在 Windows / Linux / macOS 直接运行。

3. 依赖回退：Kaiwu SDK、sksurv 均为可选依赖，缺失时脚本自动回退本地模拟退火 / 随机森林回归，不影响主流程产出。

4. 运行时间：BP-Cox 与全癌种主实验耗时较长，建议 GPU；brca_validation.py 多 seed 验证为最耗时环节。

5. 参数口径：论文“补充验证采用快速参数配置（CV=2、FullModel 上限 500、Fusion 权重 6 点）”对应癌症2_最终修改版.py 等快速脚本；与core_qubo_fusion.py 主实验（CV=3、21 权重点）的结果不可直接比较绝对值，需按论文口径分开解读。

6. 结果复现：主实验随机种子统一为 42；brca_validation.py 另有 12 个 seed 做均值与方差。

7. 编码问题：癌症2_修改后.py 为 GBK 编码（头部声明 utf-8 会导致乱码），功能已被core_qubo_fusion.py 覆盖，不建议继续使用。

8. 表2 统计缺口：论文表 2 的“零重叠对数、基因并集”未在任何脚本中直接输出，需在运行three_extension_analyses.py 问题 2 后，对产生的各 cohort QUBO 基因文件做一次独立汇总统计。
