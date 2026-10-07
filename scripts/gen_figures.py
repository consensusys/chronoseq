#!/usr/bin/env python3
"""
Generate figures from benchmark CSV data.

Produces:
  - fig_throughput_lanes.pdf
  - fig_latency_percentiles.pdf
  - fig_fairness_jain.pdf
  - fig_throughput_blocksize.pdf

Usage:
    python scripts/gen_figures.py [--data-dir data/] [--output-dir figures/]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def set_style():
    """Publication-quality plot style."""
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "figure.figsize": (5, 3.5),
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def fig_throughput_lanes(data_dir: str, out_dir: str):
    """Throughput under varying number of lanes."""
    df = pd.read_csv(f"{data_dir}/throughput_by_lanes.csv")
    fig, ax = plt.subplots()
    ax.plot(df["lanes"], df["ChronoSeq_TPS"], "o-", linewidth=1.2, label="ChronoSeq")
    ax.plot(df["lanes"], df["SingleLeader_TPS"], "s--", linewidth=1.2, label="Single-Leader Baseline")
    ax.set_xlabel("Lanes")
    ax.set_ylabel("Throughput (TPS)")
    ax.legend(frameon=False)
    fig.savefig(f"{out_dir}/fig_throughput_lanes.pdf")
    fig.savefig(f"{out_dir}/fig_throughput_lanes.png")
    plt.close(fig)
    print(f"  → {out_dir}/fig_throughput_lanes.pdf")


def fig_latency_percentiles(data_dir: str, out_dir: str):
    """End-to-end latency percentiles (p50, p95, p99)."""
    df = pd.read_csv(f"{data_dir}/latency_percentiles.csv")
    x = np.arange(len(df))
    width = 0.30
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, df["ChronoSeq_ms"], width, label="ChronoSeq", hatch="//", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, df["SingleLeader_ms"], width, label="Single-Leader", hatch="..", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["percentile"])
    ax.set_ylabel("Latency (ms)")
    ax.legend(frameon=False)
    fig.savefig(f"{out_dir}/fig_latency_percentiles.pdf")
    fig.savefig(f"{out_dir}/fig_latency_percentiles.png")
    plt.close(fig)
    print(f"  → {out_dir}/fig_latency_percentiles.pdf")


def fig_fairness(data_dir: str, out_dir: str):
    """Fairness evaluation using Jain's Index across workloads."""
    df = pd.read_csv(f"{data_dir}/fairness_jain.csv")
    x = np.arange(len(df))
    width = 0.30
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, df["ChronoSeq_Jain"], width, label="ChronoSeq", hatch="--", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, df["SingleLeader_Jain"], width, label="Single-Leader", hatch="xx", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["workload"])
    ax.set_ylabel("Jain's Index")
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False)
    fig.savefig(f"{out_dir}/fig_fairness_jain.pdf")
    fig.savefig(f"{out_dir}/fig_fairness_jain.png")
    plt.close(fig)
    print(f"  → {out_dir}/fig_fairness_jain.pdf")


def fig_throughput_blocksize(data_dir: str, out_dir: str):
    """Throughput under varying micro-block sizes."""
    df = pd.read_csv(f"{data_dir}/throughput_by_blocksize.csv")
    fig, ax = plt.subplots()
    ax.plot(df["block_kib"], df["ChronoSeq_TPS"], "^-", linewidth=1.2, label="ChronoSeq")
    ax.plot(df["block_kib"], df["SingleLeader_TPS"], "D--", linewidth=1.2, label="Single-Leader Baseline")
    ax.set_xlabel("Micro-block Size (KiB)")
    ax.set_ylabel("Throughput (TPS)")
    ax.legend(frameon=False)
    fig.savefig(f"{out_dir}/fig_throughput_blocksize.pdf")
    fig.savefig(f"{out_dir}/fig_throughput_blocksize.png")
    plt.close(fig)
    print(f"  → {out_dir}/fig_throughput_blocksize.pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate figures from CSV benchmark data")
    parser.add_argument("--data-dir", default="data", help="Directory with CSV files")
    parser.add_argument("--output-dir", default="figures", help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_style()

    print("Generating figures...")
    fig_throughput_lanes(args.data_dir, args.output_dir)
    fig_latency_percentiles(args.data_dir, args.output_dir)
    fig_fairness(args.data_dir, args.output_dir)
    fig_throughput_blocksize(args.data_dir, args.output_dir)
    print(f"✓ All figures saved to {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
