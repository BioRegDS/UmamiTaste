#!/usr/bin/env python3
"""
14_make_matched_pair_figure_v3.py

Final publication-ready matched-pair IG figure.

Changes from v2:
- larger molecular panels
- reduced unused whitespace
- phosphate annotation moved away from molecule and linked with an arrow
- q-value formatting harmonized
- panel spacing refined
"""

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def signed_color(v, vmax):
    if vmax <= 0:
        return (1, 1, 1)
    x = np.clip(v / vmax, -1, 1)
    if x >= 0:
        return (1.0, 1.0 - 0.78*x, 1.0 - 0.78*x)
    a = abs(x)
    return (1.0 - 0.78*a, 1.0 - 0.78*a, 1.0)


def draw_molecule(smiles, atom_df, specific_label, width=1100, height=600):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    atom_df = atom_df.sort_values("atom_index")
    vals = {int(r.atom_index): float(r.signed_ig)
            for r in atom_df.itertuples()}
    vmax = max(abs(v) for v in vals.values()) if vals else 1.0
    colors = {i: signed_color(v, vmax) for i, v in vals.items()}

    radii = {}
    specific = []
    for r in atom_df.itertuples():
        i = int(r.atom_index)
        is_specific = r.mcs_region == specific_label
        radii[i] = 0.47 if is_specific else 0.27
        if is_specific:
            specific.append(i)

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    opts.atomHighlightsAreCircles = True
    opts.fillHighlights = True

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(colors.keys()),
        highlightAtomColors=colors,
        highlightAtomRadii=radii
    )
    drawer.FinishDrawing()

    return Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB"), specific


def paired_panel(ax, a, b, ylabel, q, title):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    for x, y in zip(a, b):
        ax.plot([0, 1], [x, y], color="0.74", alpha=0.62,
                linewidth=0.9, zorder=1)

    ax.scatter(np.zeros(len(a)), a, s=31, color="#377eb8",
               edgecolor="white", linewidth=0.35, zorder=3)
    ax.scatter(np.ones(len(b)), b, s=31, color="#e6550d",
               edgecolor="white", linewidth=0.35, zorder=3)

    ma, mb = np.median(a), np.median(b)
    ax.plot([-0.12, 0.12], [ma, ma], color="#377eb8",
            linewidth=3.2, solid_capstyle="butt")
    ax.plot([0.88, 1.12], [mb, mb], color="#e6550d",
            linewidth=3.2, solid_capstyle="butt")

    ax.set_xlim(-0.22, 1.22)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Umami-specific", "Non-umami-specific"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12.2)

    qtxt = "FDR q < 0.001" if q < 0.001 else f"FDR q = {q:.4f}"
    ax.text(0.5, 0.975, qtxt,
            transform=ax.transAxes, ha="center", va="top", fontsize=9.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9.6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary",
        default="data/ig/matched_pairs/matched_pair_ig_summary.csv")
    p.add_argument("--atoms",
        default="data/ig/matched_pairs/matched_pair_atom_annotations.csv")
    p.add_argument("--stats",
        default="data/ig/matched_pairs/matched_pair_group_stats.csv")
    p.add_argument("--output-dir",
        default="data/ig/matched_pairs/figure_v3")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    s = pd.read_csv(args.summary)
    atoms = pd.read_csv(args.atoms)
    stats = pd.read_csv(args.stats)

    def q_for(name):
        z = stats.loc[stats["comparison"] == name, "wilcoxon_q"]
        if len(z) != 1:
            raise ValueError(f"Could not uniquely find q-value for {name}")
        return float(z.iloc[0])

    q_a = q_for("umami_specific_signed_sum_vs_nonumami_specific_signed_sum")
    q_b = q_for("umami_specific_abs_fraction_vs_nonumami_specific_abs_fraction")

    rep = s[
        (s["positive_name"] == "Adenosine 5'-monophosphate") &
        (s["negative_name"] == "Adenosine")
    ]
    if len(rep) != 1:
        raise ValueError("AMP-Adenosine representative pair not uniquely found.")
    rep = rep.iloc[0]
    pid = rep["pair_id"]

    aa = atoms[atoms["pair_id"] == pid]
    amp_atoms = aa[aa["pair_role"] == "umami"]
    ado_atoms = aa[aa["pair_role"] == "nonumami"]

    amp_img, amp_specific = draw_molecule(
        rep["umami_smiles"], amp_atoms, "umami_specific")
    ado_img, ado_specific = draw_molecule(
        rep["nonumami_smiles"], ado_atoms, "nonumami_specific")

    fig = plt.figure(figsize=(11.8, 8.8))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[0.86, 1.14],
        hspace=0.22,
        wspace=0.26,
        left=0.07, right=0.98, top=0.91, bottom=0.09
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    paired_panel(
        axA,
        s["umami_specific_signed_sum"],
        s["nonumami_specific_signed_sum"],
        "Signed IG sum",
        q_a,
        "A  Pair-specific signed attribution"
    )

    paired_panel(
        axB,
        s["umami_specific_abs_fraction"],
        s["nonumami_specific_abs_fraction"],
        "Fraction of total |IG|",
        q_b,
        "B  Attribution in pair-specific atoms"
    )

    axC.imshow(amp_img)
    axC.axis("off")
    axC.set_title(
        "C  Adenosine 5'-monophosphate (umami)\n"
        f"Specific signed IG = {rep['umami_specific_signed_sum']:.2f}; "
        f"specific |IG| fraction = {rep['umami_specific_abs_fraction']:.1%}",
        loc="left", fontweight="bold", fontsize=10.5, pad=3
    )

    # Arrowed annotation kept outside the molecule core.
    axC.annotate(
        "Pair-specific\nphosphate group",
        xy=(0.80, 0.47),
        xycoords="axes fraction",
        xytext=(0.73, 0.14),
        textcoords="axes fraction",
        ha="center", va="center",
        fontsize=9.2, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28",
                  facecolor="white", edgecolor="0.35", linewidth=0.8),
        arrowprops=dict(arrowstyle="->", color="0.25", linewidth=1.0)
    )

    axD.imshow(ado_img)
    axD.axis("off")
    axD.set_title(
        "D  Adenosine (non-umami)\n"
        "No pair-specific atoms",
        loc="left", fontweight="bold", fontsize=10.5, pad=3
    )

    fig.suptitle(
        "Integrated Gradients highlight local structural features "
        "associated with umami prediction",
        fontsize=14.4, fontweight="bold", y=0.965
    )

    fig.text(
        0.5, 0.025,
        "Atom color represents signed IG (blue, negative; white, near zero; "
        "red, positive). Larger highlights indicate pair-specific atoms. "
        "Horizontal bars indicate medians.",
        ha="center", fontsize=8.8
    )

    for ext in ("png", "pdf", "svg"):
        fn = out / f"matched_pair_ig_figure_v3.{ext}"
        fig.savefig(
            fn,
            dpi=400 if ext == "png" else None,
            bbox_inches="tight",
            facecolor="white"
        )

    plt.close(fig)

    print("Done.")
    print(f"Pairs: {len(s)}")
    print(f"Panel A FDR q: {q_a:.8g}")
    print(f"Panel B FDR q: {q_b:.8g}")
    print(f"Representative pair_id: {pid}")
    print(f"AMP pair-specific atom indices: {amp_specific}")
    print(f"Adenosine pair-specific atom indices: {ado_specific}")
    print(f"Output directory: {out}")


if __name__ == "__main__":
    main()
