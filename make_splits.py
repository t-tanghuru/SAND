"""Build the subject-level CN/AD split used for the 3T AnoDDPM experiments.

CN = the 3T scans kept from CN_MRI_raw_486subjects (data/v2) + CN_new_MRI_raw_919subjects (data/v2_CN_new919).
AD = data/v2_AD.

Rules (decided 2026-09-15):
  * CN test keeps every subject that was in the original v2 test split (whether their scan is the original 3T one or a
    new 3T scan that replaced a 1.5T one), so older models can be compared on untouched subjects.
  * The remaining CN subjects are allocated to test / val / train / reserve by systematic sampling within
    age x sex x scan-era strata, so every split has a similar mix. `reserve` is held back for the later
    "more training data" experiment and must never be used for validation or testing.
  * AD val/test reproduces the earlier evaluation exactly: tags sorted, RandomState(2026) permutation, first 48 = val.

Writes one CSV with tag, group, split, root (the preprocessing output dir holding volumes/ and nifti/) and covariates.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(REPO / "data" / "splits" / "split_3T_cn1102_ad328.csv"))
    p.add_argument("--n_test", type=int, default=152, help="CN test size including the fixed original test subjects")
    p.add_argument("--n_val", type=int, default=100)
    p.add_argument("--n_train", type=int, default=500)
    p.add_argument("--n_ad_val", type=int, default=48)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def era(year):
    return pd.cut(year, [2004, 2010, 2016, 2021, 2030], labels=["2005-10", "2011-16", "2017-21", "2022-"]).astype(str)


def load(root, participants, group):
    s = pd.read_csv(REPO / root / "subjects.csv")
    p = pd.read_csv(REPO / participants, encoding="utf-8-sig")  # the CN486 export has a BOM
    p["tag"] = p.subject_id + "_" + p.image_id
    keep = ["tag", "age", "sex", "study_date"] + [c for c in ["original_v2_split"] if c in p]
    d = s.drop(columns=["split"]).merge(p[keep], on="tag", how="left", validate="one_to_one")
    assert d.study_date.notna().all(), f"participants metadata missing for some subjects in {root}"
    if d.age.isna().any():  # e.g. 035_S_6160 has no age in the CN486 export; it lands in an "unknown age" stratum
        print(f"warning: no age for {d.loc[d.age.isna(), 'tag'].tolist()} in {participants}")
    d["group"], d["root"] = group, str(REPO / root)
    d["year"] = d.study_date.str[:4].astype(int)
    return d, s


def allocate(pool, sizes, rng):
    """Systematic allocation along a strata-sorted, shuffled order: each split gets ~its share of every stratum."""
    pool = pool.assign(_r=rng.rand(len(pool))).sort_values(["stratum", "_r"]).reset_index(drop=True)
    names, counts = list(sizes), np.array(list(sizes.values()), dtype=float)
    assert counts.sum() == len(pool), (counts.sum(), len(pool))
    target, given, labels = counts / counts.sum(), np.zeros(len(names)), []
    for i in range(len(pool)):
        k = int(np.argmax(target * (i + 1) - given))  # largest remainder
        labels.append(names[k])
        given[k] += 1
    return pool.assign(split=labels).drop(columns="_r")


def main():
    a = parse_args()
    rng = np.random.RandomState(a.seed)
    old, old_s = load("data/v2", "CN_MRI_raw_486subjects/participants.csv", "CN")
    old["was_test"] = old.tag.map(dict(zip(old_s.tag, old_s.split))) == "test"
    new, _ = load("data/v2_CN_new919", "data/CN_new_MRI_raw_919subjects/participants.csv", "CN")
    new["was_test"] = new.original_v2_split == "test"
    cn = pd.concat([old, new], ignore_index=True)
    assert cn.subject_id.is_unique, "a CN subject appears twice"
    cn["stratum"] = (pd.cut(cn.age, [0, 65, 70, 75, 80, 200]).astype(str).replace("nan", "age?") + "|" + cn.sex + "|"
                     + era(cn.year))

    fixed = cn[cn.was_test].assign(split="test")
    pool = cn[~cn.was_test]
    n_reserve = len(pool) - (a.n_test - len(fixed)) - a.n_val - a.n_train
    assert n_reserve >= 0, "not enough CN subjects for the requested split sizes"
    rest = allocate(pool, {"test": a.n_test - len(fixed), "val": a.n_val, "train": a.n_train, "reserve": n_reserve}, rng)
    cn = pd.concat([fixed, rest], ignore_index=True)

    ad, _ = load("data/v2_AD", "data/AD_MRI_raw_328subjects/participants.csv", "AD")
    ad = ad.sort_values("tag").reset_index(drop=True)
    perm = np.random.RandomState(2026).permutation(len(ad))  # identical to the earlier sand_eval_ad.py AD split
    ad["split"] = "test"
    ad.loc[perm[:a.n_ad_val], "split"] = "val"

    cols = ["tag", "subject_id", "image_id", "group", "split", "root", "age", "sex", "year", "hippo_zmin", "hippo_zmax",
            "hippo_zcenter", "hippo_voxels", "clip_lo", "clip_hi"]
    out = pd.concat([cn[cols + ["was_test"]], ad[cols].assign(was_test=False)], ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    print(pd.crosstab(out.group + " " + out.split, "n").to_string())
    print(out.groupby(["group", "split"]).agg(age=("age", "mean"), female=("sex", lambda s: (s == "F").mean()),
                                              year=("year", "median")).round(2).to_string())
    print(f"fixed original CN test subjects: {len(fixed)} | wrote {a.out}")


if __name__ == "__main__":
    main()
