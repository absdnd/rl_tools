'''
A script tp plot confidence intervals for a group of runs based on a specific criteria

Usage:
    python plot_ci.py --group-name "Adaptive" --metric_name "reward" --filter-criteria "algo.lr"

This script will extract all runs from the group "Adaptive" and group them based on the filter criteria "algo.lr"
Example: 
    group_runs = {
        0.001: runs_df_0.001,
        0.01: runs_df_0.01,
        0.1: runs_df_0.1
    }

bootstrap_ci: Will find confidence interval of mean (95% CI) using sampling with replacement 
'''

import numpy as np 
import torch
import wandb
import argparse
import pandas as pd
import matplotlib.pyplot as plt 


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-name", type=str, help="Group name for the sweep")
    parser.add_argument("--metric_name", type=str, default="reward",  help="Metric name to extract")
    parser.add_argument('--filter-criteria', type=str, default='algo.lr', help="Filter criteria for the group")
    return parser

def extract_nested_v_from_dict(D: dict, k_list):
    D_v = {k: v for k, v in D.items()}
    for k in k_list:
        D_v = D_v.get(k)
    return D_v

def boostrap_ci(run_df, n_samples=100, alpha=0.95):
    run_df.fillna(0, inplace=True)
    run_array = np.array(run_df)
    total_samples, group_n = run_array.shape[0], run_array.shape[1]
    
    samples = np.random.choice(group_n, size=(total_samples, n_samples, group_n), replace=True)
    run_array = np.expand_dims(run_array, axis=1).repeat(n_samples, axis=1)
    D_tensor = run_array.copy()
    np.put_along_axis(D_tensor, samples, run_array, axis=-1)

    means = np.mean(D_tensor, axis=-1)

    lower_list, upper_list = [], []
    for i in range(means.shape[0]):
        lower = np.percentile(means[i], 100 * (1-alpha)/2)
        upper = np.percentile(means[i], 100 * (1+alpha)/2)
        lower_list.append(lower)
        upper_list.append(upper)

    
    return np.array(lower_list), np.array(upper_list)


'''
Given a run group and project name, group the runs based on a specific criteria
For example: group_name: "Adaptive", filter_criteria: "algo.lr"

# group_runs: {
"0.001": runs_df_0.001,
...

"0.1": runs_df_0.1
}
'''

def extract_from_wb(group_name, metric_name, filter_criteria):

    api = wandb.Api()
    
    runs = api.runs(
        path = "absdnd/discrete_ir", 
        filters = {
            "config.log.group_name": group_name,
            }
        )
    
    filter_list = filter_criteria.split(".")

    group_runs = {}
    for run_v in runs:

        run_history = run_v.history()
        metric_df = run_history.get(metric_name)
        
        filter_v = extract_nested_v_from_dict(run_v.config, filter_list)
        # breakpoint()
        if metric_df is None:
            continue
        df = pd.DataFrame({run_v.name: metric_df})

        if filter_v not in group_runs:
            group_runs[filter_v] = df
        else:
            group_runs[filter_v] = pd.concat([group_runs[filter_v], df], axis=1)

    # Grouped data frame based on common value 
    group_df = {}
    for filter_v, run_df in group_runs.items():
        lower, upper = boostrap_ci(run_df)
        mean = np.mean(run_df.values, axis=1)
        group_df[filter_v] = pd.DataFrame({
            "mean": mean,
            "lower": lower,
            "upper": upper
        })

    return group_df

def plot_grouped_runs(group_k, filter_criteria, grouped_runs):
    for param_k, param_v in grouped_runs.items():
        mean = param_v["mean"]
        lower = param_v["lower"]
        upper = param_v["upper"]


    fig, ax = plt.subplots()
    ax.set_title(f"Grouped runs for {param_k}")
    ax.set_xlabel("Number of Interaction Steps x 1000")
    ax.set_ylabel(filter_criteria)
    ax.plot(mean, label="Mean")
    ax.fill_between(range(len(mean)), lower, upper, alpha=0.2)
    ax.legend()
    fig.savefig(f"{group_k}:{param_k}.png")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    group_df = extract_from_wb(args.group_name, args.metric_name, filter_criteria=args.filter_criteria)
    plot_grouped_runs(args.group_name, args.filter_criteria, group_df)