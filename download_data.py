# -*- coding: utf-8 -*-
"""
Download the SurvivalML pancancer dataset (mrna.rda files) from Synapse.

NOTE: The original download script (癌症.py) was NOT included in the source
archive. This file was reconstructed from the description in the code usage
report ("QUBO-Fusion 代码使用报告"): it downloads the SurvivalML project
dataset (Synapse ID syn58922557) into a local directory, one subdirectory per
cancer type, and skips files that have already been downloaded.

Usage:
    python download_data.py --token <SYNAPSE_TOKEN> [--synapse_id syn58922557] [--output data]

Prerequisites:
    1. Register at https://www.synapse.org and create a personal access token.
    2. pip install synapseclient synapseutils
"""

import os
import sys
import argparse
import params


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download SurvivalML mrna.rda data from Synapse"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("SYNAPSE_TOKEN", ""),
        help="Synapse personal access token (or set SYNAPSE_TOKEN env var)",
    )
    parser.add_argument(
        "--synapse_id",
        default=params.SYNAPSE_ID,
        help="SurvivalML project / folder ID on Synapse",
    )
    parser.add_argument(
        "--output",
        dest="download_dir",
        default=params.DATA_ROOT,
        help="Local directory to store the downloaded data",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    if not args.token:
        print(
            "[ERROR] No Synapse token provided. "
            "Pass --token <TOKEN> or set the SYNAPSE_TOKEN environment variable."
        )
        print("Create a token at https://www.synapse.org/#!Synapse: -> Settings -> Personal Access Tokens")
        sys.exit(1)

    try:
        import synapseclient
        import synapseutils
    except ImportError:
        print("[ERROR] Missing dependencies. Run: pip install synapseclient synapseutils")
        sys.exit(1)

    os.makedirs(args.download_dir, exist_ok=True)

    syn = synapseclient.Synapse()
    syn.login(authToken=args.token)
    print(f"Logged into Synapse. Downloading {args.synapse_id} -> {args.download_dir}")

    # Download the whole folder tree; already-downloaded files are skipped.
    files = synapseutils.syncFromSynapse(syn, args.synapse_id, path=args.download_dir)
    count = 0
    for f in files:
        if isinstance(f, synapseclient.File):
            count += 1
    print(f"Done. {count} file(s) synced to {args.download_dir}")

    print(
        "Expected layout: <output>/<CANCER>/mrna.rda, e.g. data/BRCA/mrna.rda, "
        "which is what core_qubo_fusion.py --data_root expects."
    )


if __name__ == "__main__":
    main()
