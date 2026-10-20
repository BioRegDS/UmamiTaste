#!/usr/bin/env python3
"""
18_make_study_design_figure.py

Figure 1: Study design for umami taste prediction

Panel A: Dataset curation
Panel B: Three evaluation datasets
Bottom: Common scaffold-disjoint 5-fold cross-validation

Outputs:
  figure1_study_design.png
  figure1_study_design.pdf
  figure1_study_design.svg
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def add_box(
    ax, x, y, w, h, text,
    fontsize=10,
    linewidth=1.2,
    rounding=0.02,
    fontweight="normal",
    facecolor="white",
):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        linewidth=linewidth,
        edgecolor="black",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
    )
    return box


def add_arrow(ax, x1, y1, x2, y2, linewidth=1.3):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=linewidth,
        color="black",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        default="data/figures/figure1",
    )
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.5, 6.8))
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[0.82, 1.35],
        wspace=0.12,
        left=0.04,
        right=0.98,
        bottom=0.14,
        top=0.86,
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    for ax in (axA, axB):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ---------------------------------------------------------
    # Panel A: Dataset curation
    # ---------------------------------------------------------
    axA.set_title(
        "A  Dataset curation",
        loc="left",
        fontweight="bold",
        fontsize=13,
        pad=10,
    )

    add_box(
        axA, 0.18, 0.76, 0.64, 0.13,
        "ChemTastesDB\n4,078 records",
        fontsize=11,
        fontweight="bold",
    )

    add_arrow(axA, 0.50, 0.76, 0.50, 0.68)

    add_box(
        axA, 0.18, 0.55, 0.64, 0.13,
        "Umami-labeled compounds\n220",
        fontsize=11,
    )

    add_arrow(axA, 0.50, 0.55, 0.50, 0.47)

    add_box(
        axA, 0.10, 0.31, 0.80, 0.16,
        "Low-molecular-weight curation\n"
        "Peptide / mixture exclusion\n"
        "Salt standardization / duplicate merging",
        fontsize=10,
    )

    add_arrow(axA, 0.50, 0.31, 0.50, 0.23)

    add_box(
        axA, 0.18, 0.08, 0.64, 0.15,
        "Final umami set\n74 compounds",
        fontsize=11,
        fontweight="bold",
    )

    # ---------------------------------------------------------
    # Panel B: Three evaluation datasets
    # ---------------------------------------------------------
    axB.set_title(
        "B  Three evaluation datasets",
        loc="left",
        fontweight="bold",
        fontsize=13,
        pad=10,
    )

    add_box(
        axB, 0.35, 0.76, 0.30, 0.13,
        "74 Umami compounds",
        fontsize=11,
        fontweight="bold",
    )

    # Branch points
    branch_y = 0.66
    add_arrow(axB, 0.50, 0.76, 0.50, branch_y)
    axB.plot([0.16, 0.84], [branch_y, branch_y], linewidth=1.3, color="black")

    centers = [0.17, 0.50, 0.83]
    titles = ["Strict", "Full primary", "Physchem-matched"]
    neg_counts = ["209 tasteless", "2,320 non-umami", "74 non-umami"]
    purposes = [
        "Taste-specific\ncontrol",
        "Broad chemical\nspace",
        "Basic property-\nmatched control",
    ]

    for cx, title, neg, purpose in zip(
        centers, titles, neg_counts, purposes
    ):
        add_arrow(axB, cx, branch_y, cx, 0.57)

        add_box(
            axB, cx - 0.13, 0.44, 0.26, 0.13,
            title,
            fontsize=10.5,
            fontweight="bold",
        )

        add_box(
            axB, cx - 0.13, 0.25, 0.26, 0.13,
            f"74 umami\nvs\n{neg}",
            fontsize=9.5,
        )

        add_box(
            axB, cx - 0.13, 0.07, 0.26, 0.12,
            purpose,
            fontsize=9.2,
        )

        add_arrow(axB, cx, 0.44, cx, 0.38)
        add_arrow(axB, cx, 0.25, cx, 0.19)

    # ---------------------------------------------------------
    # Shared evaluation strip
    # ---------------------------------------------------------
    fig.text(
        0.50,
        0.075,
        "All model evaluations: Murcko scaffold-disjoint 5-fold cross-validation",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
    )

    fig.suptitle(
        "Construction of datasets for umami taste prediction",
        fontsize=15,
        fontweight="bold",
        y=0.95,
    )

    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            outdir / f"figure1_study_design.{ext}",
            dpi=400 if ext == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.close(fig)

    print("Done.")
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()
