#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GROUPS=["carboxylic_acid","primary_amine","carbonyl","aromatic_ring_atom",
        "heteroaromatic_atom","hydroxyl","ether"]
DISPLAY={"carboxylic_acid":"Carboxylic acid","primary_amine":"Primary amine",
"carbonyl":"Carbonyl","aromatic_ring_atom":"Aromatic ring",
"heteroaromatic_atom":"Heteroaromatic","hydroxyl":"Hydroxyl","ether":"Ether"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--comparison",default="data/ig/functional_group_compare/functional_group_comparison.csv")
    p.add_argument("--per-molecule",default="data/ig/functional_group_compare/functional_group_per_molecule_all.csv")
    p.add_argument("--output-dir",default="data/figures/figure3_v2")
    a=p.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    comp=pd.read_csv(a.comparison); per=pd.read_csv(a.per_molecule)
    sel=comp[comp.functional_group.isin(GROUPS)].copy()
    sel["display"]=sel.functional_group.map(DISPLAY)
    sel["_order"]=sel.functional_group.map({g:i for i,g in enumerate(GROUPS)})
    sel=sel.sort_values("_order")

    fig=plt.figure(figsize=(13.8,5.2))
    gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.12],wspace=.43,
                        left=.07,right=.985,bottom=.18,top=.79)
    axA,axB,axC=[fig.add_subplot(gs[0,i]) for i in range(3)]

    # A prevalence
    d=sel.iloc[::-1]; y=np.arange(len(d)); v=d.prevalence_difference.to_numpy()
    axA.axvline(0,lw=1,alpha=.55)
    for yy,x in zip(y,v): axA.plot([0,x],[yy,yy],lw=1.2,alpha=.65)
    axA.scatter(v,y,s=44,zorder=3)
    axA.set_yticks(y); axA.set_yticklabels(d.display,fontsize=9.3)
    axA.set_xlabel("Prevalence difference\n(umami − non-umami)",fontsize=9.5)
    axA.set_title("A  Functional-group prevalence",loc="left",fontweight="bold",fontsize=12,pad=14)
    for yy,(_,r) in enumerate(d.iterrows()):
        if r.fisher_q<.05: axA.text(r.prevalence_difference,yy+.22,"*",ha="center",fontsize=11)
    axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False); axA.grid(axis="x",alpha=.17)

    # B signed IG
    d=sel.iloc[::-1]; y=np.arange(len(d)); v=d.signed_cliffs_delta.to_numpy()
    axB.axvline(0,lw=1,alpha=.55)
    for yy,x in zip(y,v): axB.plot([0,x],[yy,yy],lw=1.2,alpha=.65)
    axB.scatter(v,y,s=44,zorder=3)
    axB.set_yticks(y); axB.set_yticklabels(d.display,fontsize=9.3)
    axB.set_xlabel("Cliff's delta for signed IG\n(umami vs non-umami)",fontsize=9.5)
    axB.set_title("B  Context-dependent attribution",loc="left",fontweight="bold",fontsize=12,pad=14)
    for yy,(_,r) in enumerate(d.iterrows()):
        if r.signed_mw_q<.05: axB.text(r.signed_cliffs_delta,yy+.22,"*",ha="center",fontsize=11)
    axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False); axB.grid(axis="x",alpha=.17)

    # C carboxylic acid
    c=per[per.functional_group=="carboxylic_acid"]
    u=c[c.is_umami_positive==1].signed_ig_sum.astype(float).to_numpy()
    n=c[c.is_umami_positive==0].signed_ig_sum.astype(float).to_numpy()
    parts=axC.violinplot([u,n],positions=[0,1],widths=.58,showmeans=False,showmedians=False,showextrema=False)
    for body in parts["bodies"]: body.set_alpha(.16)
    rng=np.random.default_rng(42)
    axC.scatter(rng.normal(0,.045,len(u)),u,s=25,alpha=.68,zorder=3)
    axC.scatter(1+rng.normal(0,.055,len(n)),n,s=13,alpha=.20,zorder=2)
    for x,z in [(0,np.median(u)),(1,np.median(n))]:
        axC.plot([x-.14,x+.14],[z,z],lw=3,zorder=4)
    r=comp[comp.functional_group=="carboxylic_acid"].iloc[0]
    axC.set_xticks([0,1]); axC.set_xticklabels([f"Umami\n(n={len(u)})",f"Non-umami\n(n={len(n)})"],fontsize=9.4)
    axC.set_ylabel("Molecule-level signed IG sum",fontsize=9.5)
    axC.set_title("C  Carboxylic acid",loc="left",fontweight="bold",fontsize=12,pad=14)
    axC.text(.5,.97,
             f"Prevalence: {r.umami_prevalence:.1%} vs {r.nonumami_prevalence:.1%} (FDR q={r.fisher_q:.3f})\n"
             f"Signed IG: {r.umami_signed_median:.2f} vs {r.nonumami_signed_median:.2f} (FDR q={r.signed_mw_q:.3f})",
             transform=axC.transAxes,ha="center",va="top",fontsize=8.6)
    axC.spines["top"].set_visible(False); axC.spines["right"].set_visible(False); axC.grid(axis="y",alpha=.17)

    fig.suptitle("Integrated Gradients reveal context-dependent attribution of molecular functional groups",
                 fontsize=14.5,fontweight="bold",y=.94)
    fig.text(.5,.055,"Asterisks indicate FDR-adjusted q < 0.05. In Panel C, points represent individual molecules and horizontal bars indicate medians.",
             ha="center",fontsize=8.7)
    for ext in ("png","pdf","svg"):
        fig.savefig(out/f"figure3_functional_group_v2.{ext}",dpi=400 if ext=="png" else None,
                    bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print("Done.")
    print("Output directory:",out)
    print(f"Carboxylic acid prevalence: {r.umami_prevalence:.4f} vs {r.nonumami_prevalence:.4f}; FDR q={r.fisher_q:.6g}")
    print(f"Carboxylic acid signed IG median: {r.umami_signed_median:.4f} vs {r.nonumami_signed_median:.4f}; FDR q={r.signed_mw_q:.6g}")

if __name__=="__main__":
    main()
