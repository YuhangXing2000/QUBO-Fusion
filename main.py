# -*- coding: utf-8 -*-
"""
QUBO-Fusion 主程序入口
============================================================
统一通过 main.py 的 --mode 参数运行各步骤（对齐量子开发实验室项目规范）：

    数据准备：
        python main.py --mode=data            # 从 Synapse 下载 SurvivalML 数据（需 --token）
    实验与分析（按顺序执行）：
        python main.py --mode=main            # 核心主实验（QUBO 基因选择 + RSF/BP-Cox/Fusion）
        python main.py --mode=baselines       # 强基线补充（mRMR / RSF_Importance）
        python main.py --mode=extensions      # 扩展验证（跨癌种迁移 / 跨数据集 / QUBO vs FullModel）
        python main.py --mode=brca            # BRCA 专项强基线验证（12 随机种子）
        python main.py --mode=pan             # 泛癌种汇总与基因 Jaccard 分析
        python main.py --mode=visualize       # 出版级可视化精修（可选）
        python main.py --mode=all             # 依序运行 main → baselines → extensions → pan → visualize

    绝大多数参数在 params.py 中统一调整；需要临时覆盖时，在 --mode 后追加
    对应脚本支持的命令行参数即可，例如：
        python main.py --mode=brca --n_seeds 3
        python main.py --mode=main --cancers BRCA,LUAD
"""

import argparse
import subprocess
import sys

# mode -> (脚本名, 说明)
MODES = {
    "data": ("download_data.py", "数据下载与预处理（Synapse）"),
    "main": ("core_qubo_fusion.py", "核心主实验"),
    "baselines": ("strong_baselines.py", "强基线补充"),
    "extensions": ("three_extension_analyses.py", "扩展验证"),
    "brca": ("brca_validation.py", "BRCA 专项验证"),
    "pan": ("pan_cancer_summary.py", "泛癌种汇总"),
    "visualize": ("enhanced_visualization.py", "增强可视化"),
}

# --mode=all 的执行顺序
ALL_PIPELINE = ["main", "baselines", "extensions", "pan", "visualize"]


def run_script(script, extra_args):
    cmd = [sys.executable, script] + extra_args
    print("\n" + "=" * 78)
    print("  [QUBO-Fusion] 运行: " + " ".join(cmd))
    print("=" * 78)
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="QUBO-Fusion 主程序入口：统一调度各实验步骤"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=list(MODES.keys()) + ["all"],
        help="运行模式：" + " / ".join(list(MODES.keys()) + ["all"]),
    )
    parser.add_argument(
        "extra",
        nargs="*",
        help="透传给子脚本的额外命令行参数（覆盖 params.py 默认值）",
    )
    args = parser.parse_args()

    if args.mode == "all":
        # 依序运行全部分析步骤（data 需要 Synapse token，不包含在 all 中）
        rc = 0
        for mode in ALL_PIPELINE:
            script, _ = MODES[mode]
            code = run_script(script, args.extra)
            if code != 0:
                print(f"\n[QUBO-Fusion] {mode} 失败（退出码 {code}），流水线终止。")
                sys.exit(code)
            rc = code
        print("\n[QUBO-Fusion] 全部步骤完成。")
        sys.exit(rc)
    else:
        script, desc = MODES[args.mode]
        print(f"[QUBO-Fusion] 模式 {args.mode}：{desc}")
        code = run_script(script, args.extra)
        sys.exit(code)


if __name__ == "__main__":
    main()
