import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from generative.networks.schedulers import PNDMScheduler
from monai.config import print_config
from monai.utils import set_determinism
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Location for models.")
    parser.add_argument("--model_name", help="Name of model.")
    parser.add_argument("--max_t", type=int, default=1000, help="Maximum T to consider.")
    parser.add_argument("--min_t", type=int, default=0, help="Minimum T to consider.")
    parser.add_argument("--t_skip", type=int, default=1, help="Only use every n reconstructions.")

    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    all_results_dict = {}
    model = args.model_name
    all_results_dict[model] = {}
    run_dir = Path(args.output_dir) / model
    print(f"Run directory: {str(run_dir)}")

    out_dir = run_dir / "ood"
    out_dir.mkdir(exist_ok=True)
    results_df_val = pd.read_csv(out_dir / "results_val.csv")
    results_df_val.drop_duplicates(subset=["filename", "t"], keep="first", inplace=True)
    all_t_values = results_df_val["t"].unique()
    MAX_T = args.max_t
    MIN_T = args.min_t
    T_SKIP_FACTOR = 1
    t_values = all_t_values[::T_SKIP_FACTOR]
    t_values = t_values[(t_values < MAX_T)]
    t_values = t_values[(MIN_T < t_values)]

    total_steps = 0
    pndm_scheduler = PNDMScheduler(num_train_timesteps=1000, skip_prk_steps=True)
    pndm_scheduler.set_timesteps(100)
    pndm_timesteps = pndm_scheduler.timesteps
    for t in t_values:
        steps_for_this_t = pndm_timesteps[pndm_timesteps <= t]
        total_steps += len(steps_for_this_t)

    plot_target = "mse"
    print(f"SETTING MAX_T to {MAX_T} and T_SKIP to {T_SKIP_FACTOR} with a total of {len(t_values)} starting points {total_steps} model evaluations")
    print(f"Plot target is {plot_target}")

    results_df_val = results_df_val[results_df_val["t"].isin(t_values)]
    results_df_val_pivot = results_df_val.pivot_table(
        index=["filename"], columns=["t"], values=[plot_target]
    )

    # Dataset-specific out-of-distribution selection
    # Extend this dict as needed for your brain MRI experiment
    mednist_datasets = dict.fromkeys(["AbdomenCT", "BreastMRI", "ChestCT", "CXR", "Hand", "HeadCT"])
    if "healthy" in model.lower() or "normal" in model.lower():
        out_data = ("anomalous",)
    elif "fashionmnist" in model:
        out_data = ("MNIST", "FashionMNIST_vflip", "FashionMNIST_hflip")
    elif "mnist" in model:
        out_data = ("FashionMNIST", "MNIST_vflip", "MNIST_hflip")
    elif "decathlon" in model or "Task01" in model:
        out_data = ("Task02", "Task03", "Task04", "Task05", "Task06", "Task07", "Task08", "Task09", "Task10")
    else:
        raise ValueError(f"Unknown dataset to select for run_dir {model}. Please add your dataset mapping here.")

    t_values = results_df_val["t"].unique()
    num_val_images = len(results_df_val["filename"].unique())

    for out_dataset in out_data:
        results_df_in = pd.read_csv(out_dir / "results_in.csv")
        results_df_out = pd.read_csv(out_dir / f"results_{out_dataset}.csv")
        results_df_in.drop_duplicates(subset=["filename", "t"], keep="first", inplace=True)
        results_df_out.drop_duplicates(subset=["filename", "t"], keep="first", inplace=True)
        results_df_in = results_df_in[results_df_in["t"].isin(t_values)]
        results_df_out = results_df_out[results_df_out["t"].isin(t_values)]
        results_df = pd.concat((results_df_in, results_df_out))

        for target in ["perceptual_difference", "mse"]:
            results_df_val_agg = (
                results_df_val.groupby(["t"])
                .agg({target: ["mean", "std"]})[target]
                .reset_index()
                .rename({"mean": f"val_mean_{target}", "std": f"val_std_{target}"}, axis=1)
            )
            results_df = results_df.merge(results_df_val_agg, on=["t"], how="left")
            results_df[f"z_score_{target}"] = (
                results_df[target] - results_df[f"val_mean_{target}"]
            ) / results_df[f"val_std_{target}"]

        num_in_images = results_df.loc[results_df["type"] == "in"]["filename"].nunique()
        num_out_images = results_df.loc[results_df["type"] == "out"]["filename"].nunique()

        target = f"z_score_{plot_target}"
        results_df_mean = results_df.groupby(["filename", "type"]).mean().reset_index()

        all_scores = results_df_mean.loc[results_df_mean["type"] == "in"][[target]].values.tolist()
        all_class = [0] * len(all_scores)
        all_scores.extend(results_df_mean.loc[results_df_mean["type"] == "out"][[target]].values.tolist())
        all_class.extend([1] * len(results_df_mean.loc[results_df_mean["type"] == "out"][[target]].values.tolist()))

        roc_score = roc_auc_score(all_class, all_scores)
        print(f"n_val={num_val_images} n_in={num_in_images} n_out={num_out_images}")
        method_name = f"Zscore_{plot_target}"
        if method_name in all_results_dict[model]:
            all_results_dict[model][method_name].extend([roc_score])
            all_results_dict[model]["ood_data"].extend([out_dataset])
        else:
            all_results_dict[model][method_name] = [roc_score]
            all_results_dict[model]["ood_data"] = [out_dataset]

    for method in [f"Zscore_{plot_target}"]:
        ood_datasets = all_results_dict[model]["ood_data"]
        scores = all_results_dict[model][method]
        for o, s in zip(ood_datasets, scores):
            print(f"AUC for {model} vs {o}: {s * 100:.1f}")
        print(f"Average AUC: {np.mean(scores) * 100:.1f}")


if __name__ == "__main__":
    args = parse_args()
    for model in args.model_name.split(","):
        args_copy = args
        args_copy.model_name = model
        main(args_copy)
