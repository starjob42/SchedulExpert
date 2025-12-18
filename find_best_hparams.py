import argparse
import json
import os
from collections import defaultdict


def load_all_data(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_hyperparams_for_cfg(all_data, method, cfg_id_str):
    """
    Given a method (e.g. 'smc') and cfg_id_str (e.g. 'smc_cfg00'),
    look into the per-instance section and return its 'hyperparams' dict.
    We just grab it from the first instance that has it, since configs
    are shared across instances.
    """
    instances = all_data.get("instances", {})
    for inst_name, inst_data in instances.items():
        methods = inst_data.get("methods", {})
        mentry = methods.get(method)
        if not mentry:
            continue
        cfgs = mentry.get("configs", {})
        centry = cfgs.get(cfg_id_str)
        if centry is not None:
            # Should contain "hyperparams"
            return centry.get("hyperparams", {})
    # If we reach here, we couldn't find it (weird, but let's not crash)
    return {}


def find_best_configs(all_data, metric_key="avg_gap_mean"):
    """
    Traverse all_data["_summary"]["per_budget"] to find, for each method and budget,
    the cfg_id with the best (smallest) metric_key (e.g. 'avg_gap_mean').

    Returns:
      best[method][beta] = {
          "cfg_id": ...,
          "metric_value": ...,
          "time_mean": ...,
          "time_std": ...,
      }
    """
    summary = all_data.get("_summary")
    if summary is None:
        raise RuntimeError(
            "No '_summary' found in JSON. "
            "Run the hyperparam_search script first so it fills _summary."
        )

    per_budget = summary.get("per_budget", {})
    best = defaultdict(dict)

    for beta_str, methods in per_budget.items():
        beta = int(beta_str)
        for method, meth_cfgs in methods.items():
            best_cfg_id = None
            best_metric_val = None
            best_time_mean = None
            best_time_std = None

            for cfg_id_str, stats in meth_cfgs.items():
                if stats.get("n_instances", 0) == 0:
                    continue
                mval = stats.get(metric_key, None)
                if mval is None:
                    continue
                # smaller gap is better
                if best_metric_val is None or mval < best_metric_val:
                    best_metric_val = mval
                    best_cfg_id = cfg_id_str
                    best_time_mean = stats.get("time_mean", None)
                    best_time_std = stats.get("time_std", None)

            if best_cfg_id is not None:
                best[method][beta] = {
                    "cfg_id": best_cfg_id,
                    "metric_value": best_metric_val,
                    "time_mean": best_time_mean,
                    "time_std": best_time_std,
                }

    return best


def main():
    parser = argparse.ArgumentParser(
        description="Extract best hyperparameters (per method, per budget) "
                    "from all_instances.json."
    )
    parser.add_argument(
        "--all_json",
        type=str,
        required=True,
        help="Path to all_instances.json produced by hyperparam_search_smc_cem.py",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="avg_gap_mean",
        choices=["min_gap_mean", "avg_gap_mean", "max_gap_mean"],
        help="Metric to minimize when choosing the best config "
             "(default: avg_gap_mean).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.all_json):
        raise FileNotFoundError(f"JSON file not found: {args.all_json}")

    all_data = load_all_data(args.all_json)
    best = find_best_configs(all_data, metric_key=args.metric)

    print(f"Best configs per method & budget (minimizing {args.metric}):\n")

    for method in sorted(best.keys()):
        print(f"=== Method: {method} ===")
        for beta in sorted(best[method].keys()):
            info = best[method][beta]
            cfg_id_str = info["cfg_id"]
            metric_val = info["metric_value"]
            tmean = info["time_mean"]
            tstd = info["time_std"]

            # Grab hyperparameters from the per-instance section
            hparams = extract_hyperparams_for_cfg(all_data, method, cfg_id_str)

            print(f"- β = {beta}")
            print(f"  cfg_id       : {cfg_id_str}")
            print(f"  {args.metric}: {metric_val:.4f}")
            if tmean is not None:
                print(f"  time_mean    : {tmean:.4f} s")
            if tstd is not None:
                print(f"  time_std     : {tstd:.4f} s")
            print(f"  hyperparams  : {hparams}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()


#python find_best_hparams.py --all_json hyper_smc_cem/all_instances.json