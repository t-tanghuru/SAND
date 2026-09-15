"""SAND: AUROC tables (with bootstrap 95% CIs) from sand_eval_ad.py subject scores.

Shards of one run (*_shardIofN_subjects.csv) are merged. Subgroups:
  all            every evaluated subject
  age-sex matched  each CN subject paired with the nearest-age AD subject of the same sex, without replacement
                   (AD outnumbers CN here, so AD is sub-sampled; Parker et al. 2025 matched controls on age and sex)
  age 70-80      both groups restricted to ages 70-80
With --meta_csv omitted the old CN486/AD328 metadata (field strength etc.) is used.

usage: python sand_analyze.py --meta_csv SPLIT.csv RUN_subjects.csv [RUN_shard0of3_subjects.csv ...]
"""
import argparse
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OLD_META = "/scratch/users/tjdnjs/sand_meta_cn_ad.csv"


def load_meta(meta_csv):
    if meta_csv:
        m = pd.read_csv(meta_csv)
        return m[["tag", "age", "sex", "year"]]
    m = pd.read_csv(OLD_META)
    m["tag"] = m.subject_id + "_" + m.image_id
    m["tesla"] = np.where((m.field_strength < 2) | (m.field_strength > 1000), "1.5T", "3T")
    m["year"] = m.study_date.str[:4].astype(int)
    return m[["tag", "age", "sex", "year", "tesla"]]


def auroc_ci(y, s, n_boot=2000, seed=0):
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(seed)
    point, boots = roc_auc_score(y, s), []
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            boots.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def age_sex_match(d, seed=0):
    """Greedy 1:1 nearest-age matching within sex, smaller group as anchor, without replacement."""
    d = d[d.age.notna()]  # one CN subject has no recorded age
    cn, ad = d[d.label == 0], d[d.label == 1]
    anchor, pool = (cn, ad) if len(cn) <= len(ad) else (ad, cn)
    anchor = anchor.sample(frac=1, random_state=seed)
    used, keep = set(), []
    for _, r in anchor.iterrows():
        cand = pool[(pool.sex == r.sex) & ~pool.tag.isin(used)]
        if cand.empty:
            continue
        j = (cand.age - r.age).abs().idxmin()
        used.add(pool.loc[j, "tag"])
        keep += [r.tag, pool.loc[j, "tag"]]
    return d[d.tag.isin(keep)]


SUBGROUPS = {
    "all": lambda d: d,
    "age-sex matched": age_sex_match,
    "age 70-80": lambda d: d[(d.age >= 70) & (d.age < 80)],
}


def main(paths, meta_csv=None):
    meta = load_meta(meta_csv)
    runs = defaultdict(list)
    for p in paths:
        runs[re.sub(r"_shard\d+of\d+", "", p.split("/")[-1].replace("_subjects.csv", ""))].append(pd.read_csv(p))
    out = []
    for name, parts in runs.items():
        d = pd.concat(parts).drop_duplicates(["tag", "lam"]).merge(meta, on="tag", how="left")
        assert d.age.notna().all(), f"missing metadata in {name}"
        for lam, g in d.groupby("lam"):
            for sg, f in SUBGROUPS.items():
                h = f(g)
                row = dict(run=name, lam=lam, subgroup=sg, n_CN=int((h.label == 0).sum()), n_AD=int((h.label == 1).sum()),
                           age_CN=round(h[h.label == 0].age.mean(), 1), age_AD=round(h[h.label == 1].age.mean(), 1))
                for score in ["hippo_mse", "brain_mse"]:
                    a, lo, hi = auroc_ci(h.label, h[score])
                    row[score] = f"{a:.3f} [{lo:.3f}-{hi:.3f}]"
                out.append(row)
    table = pd.DataFrame(out)
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", default=None)
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    main(a.paths, a.meta_csv)
