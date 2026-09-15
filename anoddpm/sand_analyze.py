"""SAND: subgroup AUROCs (all / 3T only / age 70-80 / 2011-16 scans) from sand_eval_ad.py subject scores.

usage: python sand_analyze.py sand_eval/args101_val_subjects.csv [more *_subjects.csv ...]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

META = "/scratch/users/tjdnjs/sand_meta_cn_ad.csv"  # CN/AD participants + DICOM field strength/vendor


def load_meta():
    m = pd.read_csv(META)
    m["tag"] = m.subject_id + "_" + m.image_id
    m["tesla"] = np.where((m.field_strength < 2) | (m.field_strength > 1000), "1.5T", "3T")  # 1.494 and a 15000 typo are 1.5T
    m["year"] = m.study_date.str[:4].astype(int)
    return m[["tag", "age", "sex", "year", "tesla", "vendor"]]


def auroc_ci(y, s, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    point = roc_auc_score(y, s)
    boots = []
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            boots.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


SUBGROUPS = {
    "all": lambda d: d,
    "3T only": lambda d: d[d.tesla == "3T"],
    "age 70-80": lambda d: d[(d.age >= 70) & (d.age < 80)],
    "scans 2011-16": lambda d: d[(d.year >= 2011) & (d.year <= 2016)],
}


def main(paths):
    meta = load_meta()
    out = []
    for p in paths:
        d = pd.read_csv(p).merge(meta, on="tag", how="left")
        assert d.tesla.notna().all(), f"missing metadata for some subjects in {p}"
        name = p.split("/")[-1].replace("_subjects.csv", "")
        for lam, g in d.groupby("lam"):
            for sg, f in SUBGROUPS.items():
                h = f(g)
                row = dict(run=name, lam=lam, subgroup=sg, n_CN=int((h.label == 0).sum()), n_AD=int((h.label == 1).sum()))
                for score in ["brain_mse", "hippo_mse"]:
                    a, lo, hi = auroc_ci(h.label, h[score])
                    row[score] = f"{a:.3f} [{lo:.3f}-{hi:.3f}]"
                out.append(row)
    table = pd.DataFrame(out)
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    main(sys.argv[1:])
