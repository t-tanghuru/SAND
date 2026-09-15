"""SAND: validation curves from sand_eval_ad.py subject scores (shards merged).

  lambda   AUROC vs partial-diffusion lambda for one checkpoint (the SAND analogue of AnoDDPM Fig. 4, which plotted
           Dice/IoU vs lambda for two tumour patients; AD has no voxel masks, so the curve is CN-vs-AD AUROC)
  ckpt     AUROC vs training iterations, one figure per lambda (files named ..._checkpoint_diff_epoch=E_...)

usage: python sand_plot_curves.py lambda OUT.png RUN_subjects.csv [shards...]
       python sand_plot_curves.py ckpt   OUT.png sand_eval/ckpt_curve/*_subjects.csv
"""
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sand_analyze import auroc_ci

SCORES = [("hippo_mse", "Hippocampus score (primary)", "#2a78d6"), ("brain_mse", "Whole-brain score", "#9aa3ad")]


def curve(d, key):
    rows = []
    for k, g in d.groupby(key):
        r = {key: k}
        for s, _, _ in SCORES:
            r[s], r[s + "_lo"], r[s + "_hi"] = auroc_ci(g.label, g[s])
        rows.append(r)
    return pd.DataFrame(rows).sort_values(key)


def plot(c, key, xlabel, title, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for s, label, color in SCORES:
        ax.fill_between(c[key], c[s + "_lo"], c[s + "_hi"], color=color, alpha=0.15, linewidth=0)
        ax.plot(c[key], c[s], marker="o", markersize=4, lw=2, color=color, label=label)
    best = c.loc[c["hippo_mse"].idxmax()]
    ax.annotate(f"best {best['hippo_mse']:.3f} @ {int(best[key])}", (best[key], best["hippo_mse"]),
                textcoords="offset points", xytext=(6, 8), fontsize=9)
    ax.axhline(0.5, color="#c9ced4", lw=1, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CN vs AD AUROC (validation)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(axis="y", color="#e6e8eb", lw=0.6)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(c.round(4).to_string(index=False))
    print("wrote", out)


def main():
    mode, out, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    parts = []
    for p in paths:
        d = pd.read_csv(p)
        m = re.search(r"diff_epoch=(\d+)", p)
        d["iterations"] = int(m.group(1)) * 100 if m else -1  # 1 epoch = 100 iterations in the official loop
        parts.append(d)
    d = pd.concat(parts).drop_duplicates(["tag", "lam", "iterations"])
    if mode == "lambda":
        assert d.iterations.nunique() == 1, "lambda curve expects one checkpoint"
        plot(curve(d, "lam"), "lam", "lambda (partial diffusion steps)", "Validation AUROC vs lambda", out)
    else:
        for lam, g in d.groupby("lam"):  # one figure per lambda: OUT_lam{lambda}.png
            if g.iterations.nunique() < 2:  # e.g. extra lambdas that only the 300k lambda curve evaluated
                continue
            o = out.replace(".png", f"_lam{lam}.png")
            plot(curve(g, "iterations"), "iterations", "training iterations", f"Validation AUROC vs training (lambda={lam})", o)


if __name__ == "__main__":
    main()
