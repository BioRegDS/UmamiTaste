#!/usr/bin/env python3
"""
15_make_performance_figure_v4.py

Finalized Figure 2:
A. ROC-AUC
B. PR-AUC with class-prevalence random baselines
C. Physicochemical balance after matching

v4 change:
- Adds extra horizontal margin only to Panel A so the Strict point/value label
  no longer collides visually with the y-axis tick labels.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRIC_RE = re.compile(r">\s*(roc_auc|pr_auc),([0-9.]+)±([0-9.]+)")


def parse_metrics(log_path):
    text = Path(log_path).read_text(errors="ignore")
    out = {}
    for metric, mean, sd in METRIC_RE.findall(text):
        out[metric] = (float(mean), float(sd))

    for metric in ("roc_auc", "pr_auc"):
        if metric not in out:
            raise ValueError(f"{metric} not found in {log_path}")

    return out


def pooled_smd(x1, x0):
    x1 = np.asarray(x1, dtype=float)
    x0 = np.asarray(x0, dtype=float)

    pooled = np.sqrt(
        (np.std(x1, ddof=1) ** 2 + np.std(x0, ddof=1) ** 2) / 2.0
    )

    if pooled == 0:
        return 0.0

    return (np.mean(x1) - np.mean(x0)) / pooled


def auc_panel(
    ax,
    ylabel,
    means,
    sds,
    labels,
    title,
    ymin,
    ymax,
    label_offset=0.018,
):
    x = np.arange(len(labels))

    ax.errorbar(
        x,
        means,
        yerr=sds,
        fmt="o",
        markersize=7.5,
        capsize=5,
        linewidth=1.6,
        zorder=3,
    )

    for i, (v, e) in enumerate(zip(means, sds)):
        y_text = min(ymax - 0.035, v + e + label_offset)

        ax.text(
            i,
            y_text,
            f"{v:.3f} ± {e:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.3)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel, fontsize=10)

    ax.set_title(
        title,
        loc="left",
        fontweight="bold",
        fontsize=12,
        pad=16,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.17)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--strict-log",
        default="data/logs/umami_strict_scaffold_cv/2026-10-03_09-29/"
                "log_2026-10-03_09-29.log",
    )
    p.add_argument(
        "--full-log",
        default="data/logs/umami_full_primary_scaffold_cv/2026-10-03_09-51/"
                "log_2026-10-03_09-51.log",
    )
    p.add_argument(
        "--matched-log",
        default="data/logs/umami_physchem_matched_scaffold_cv/2026-10-03_12-48/"
                "log_2026-10-03_12-48.log",
    )
    p.add_argument(
        "--matched-csv",
        default="data/model_sets/training_balanced_physchem_matched.csv",
    )
    p.add_argument(
        "--output-dir",
        default="data/figures/figure2_v4",
    )

    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            "name": "Strict",
            "label": "Strict",
            "n_pos": 74,
            "n_neg": 209,
            "log": args.strict_log,
        },
        {
            "name": "Full primary",
            "label": "Full primary",
            "n_pos": 74,
            "n_neg": 2320,
            "log": args.full_log,
        },
        {
            "name": "Physchem-matched",
            "label": "Physchem-\nmatched",
            "n_pos": 74,
            "n_neg": 74,
            "log": args.matched_log,
        },
    ]

    metric_rows = []

    for d in datasets:
        metrics = parse_metrics(d["log"])

        d["roc_mean"], d["roc_sd"] = metrics["roc_auc"]
        d["pr_mean"], d["pr_sd"] = metrics["pr_auc"]
        d["baseline"] = d["n_pos"] / (d["n_pos"] + d["n_neg"])

        metric_rows.append(
            {
                "dataset": d["name"],
                "n_umami": d["n_pos"],
                "n_nonumami": d["n_neg"],
                "roc_auc_mean": d["roc_mean"],
                "roc_auc_sd": d["roc_sd"],
                "pr_auc_mean": d["pr_mean"],
                "pr_auc_sd": d["pr_sd"],
                "random_pr_auc_baseline": d["baseline"],
            }
        )

    pd.DataFrame(metric_rows).to_csv(
        outdir / "figure2_metrics.csv",
        index=False,
    )

    matched = pd.read_csv(args.matched_csv)

    pos = matched[matched["label"] == 1]
    neg = matched[matched["label"] == 0]

    properties = [
        ("MW", "calc_MW"),
        ("LogP", "calc_LogP"),
        ("TPSA", "calc_TPSA"),
        ("HBD", "calc_HBD"),
        ("HBA", "calc_HBA"),
        ("RotB", "calc_RotB"),
    ]

    smd_rows = []

    for display, column in properties:
        smd = pooled_smd(pos[column], neg[column])

        smd_rows.append(
            {
                "property": display,
                "column": column,
                "umami_mean": pos[column].mean(),
                "nonumami_mean": neg[column].mean(),
                "smd": smd,
                "abs_smd": abs(smd),
            }
        )

    smd_df = pd.DataFrame(smd_rows)

    smd_df.to_csv(
        outdir / "figure2_physchem_smd.csv",
        index=False,
    )

    labels = [d["label"] for d in datasets]

    fig = plt.figure(figsize=(14.2, 5.35))

    gs = fig.add_gridspec(
        1,
        3,
        wspace=0.34,
        left=0.055,
        right=0.985,
        bottom=0.20,
        top=0.75,
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # ---------------------------------------------------------
    # A: ROC-AUC
    # ---------------------------------------------------------
    auc_panel(
        axA,
        "ROC-AUC",
        [d["roc_mean"] for d in datasets],
        [d["roc_sd"] for d in datasets],
        labels,
        "A  ROC-AUC",
        ymin=0.45,
        ymax=1.06,
        label_offset=0.012,
    )

    # Extra left/right margin only for Panel A.
    # This shifts the Strict point/value label away from the y-axis tick labels.
    axA.set_xlim(-0.30, 2.10)

    # ---------------------------------------------------------
    # B: PR-AUC
    # ---------------------------------------------------------
    auc_panel(
        axB,
        "PR-AUC",
        [d["pr_mean"] for d in datasets],
        [d["pr_sd"] for d in datasets],
        labels,
        "B  PR-AUC with random baseline",
        ymin=0.0,
        ymax=1.06,
        label_offset=0.012,
    )

    for i, d in enumerate(datasets):
        baseline = d["baseline"]

        axB.hlines(
            baseline,
            i - 0.23,
            i + 0.23,
            linestyles="--",
            linewidth=1.25,
            alpha=0.80,
        )

        axB.text(
            i,
            baseline + 0.028,
            f"{baseline:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
        )

    axB.text(
        0.5,
        -0.22,
        "Dashed segments indicate class-prevalence PR-AUC baselines.",
        transform=axB.transAxes,
        ha="center",
        va="top",
        fontsize=8.1,
    )

    # ---------------------------------------------------------
    # C: physicochemical balance
    # ---------------------------------------------------------
    plot_smd = smd_df.iloc[::-1].copy()

    y = np.arange(len(plot_smd))

    axC.scatter(
        plot_smd["abs_smd"],
        y,
        s=48,
        zorder=3,
    )

    for yy, value in zip(y, plot_smd["abs_smd"]):
        axC.plot(
            [0, value],
            [yy, yy],
            linewidth=1.15,
            alpha=0.65,
            zorder=2,
        )

    axC.axvline(
        0.10,
        linestyle="--",
        linewidth=1.0,
        alpha=0.40,
    )

    axC.text(
        0.105,
        0.45,
        "|SMD| = 0.10 reference",
        fontsize=7.9,
        rotation=90,
        va="bottom",
        ha="left",
        alpha=0.75,
    )

    axC.set_yticks(y)
    axC.set_yticklabels(
        plot_smd["property"],
        fontsize=9.5,
    )

    axC.set_xlabel(
        "Absolute standardized mean difference",
        fontsize=9.8,
    )

    axC.set_title(
        "C  Physicochemical balance after matching",
        loc="left",
        fontweight="bold",
        fontsize=12,
        pad=16,
    )

    axC.set_xlim(
        0,
        max(
            0.45,
            float(plot_smd["abs_smd"].max()) * 1.18,
        ),
    )

    axC.spines["top"].set_visible(False)
    axC.spines["right"].set_visible(False)
    axC.grid(axis="x", alpha=0.17)

    # ---------------------------------------------------------
    # Overall title / footnote
    # ---------------------------------------------------------
    fig.suptitle(
        "Robust prediction of umami taste under scaffold-disjoint evaluation",
        fontsize=15,
        fontweight="bold",
        y=0.94,
    )

    fig.text(
        0.5,
        0.045,
        "Points and error bars show mean ± SD from scaffold-disjoint 5-fold cross-validation.",
        ha="center",
        fontsize=8.8,
    )

    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            outdir / f"figure2_performance_v4.{ext}",
            dpi=400 if ext == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.close(fig)

    print("Done.")

    for d in datasets:
        print(
            f'{d["name"]}: '
            f'ROC-AUC {d["roc_mean"]:.4f} ± {d["roc_sd"]:.4f}; '
            f'PR-AUC {d["pr_mean"]:.4f} ± {d["pr_sd"]:.4f}; '
            f'baseline {d["baseline"]:.4f}'
        )

    print("Physicochemical |SMD|:")

    for _, row in smd_df.iterrows():
        print(
            f'  {row["property"]}: '
            f'{row["abs_smd"]:.4f}'
        )

    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()
