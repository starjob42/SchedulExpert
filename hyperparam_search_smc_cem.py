import argparse
import os
import glob
import json
import time
import random

import numpy as np
import torch
import torch.nn.functional as F  # kept for compatibility; may be unused

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving PDFs
import matplotlib.pyplot as plt
import matplotlib as mpl

from tqdm import tqdm

from utils.sampling import JobShopStates
from utils.inout import load_data
from sampling_methods import sampling, _smc_sampling_fast_improved, _cem_guided_sampling


# ---------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------
BASE_FONTSIZE = 11
SCALE = 3.4

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",

    "font.size": BASE_FONTSIZE * SCALE,
    "axes.labelsize": BASE_FONTSIZE * SCALE,
    "xtick.labelsize": BASE_FONTSIZE * SCALE * 0.9,
    "ytick.labelsize": BASE_FONTSIZE * SCALE * 0.9,
    "legend.fontsize": BASE_FONTSIZE * SCALE * 0.9,

    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,

    "figure.figsize": (5.0 * SCALE, 3.5 * SCALE),
    "legend.frameon": False,
    "figure.autolayout": True,
})

METHOD_COLORS = {
    "sampling": "#0072B2",
    "smc":      "#D55E00",
    "cem":      "#009E73",
}

METHOD_DISPLAY = {
    "sampling": "Ordinary sampling",
    "smc": "SMC",
    "cem": "CEM",
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_args(arguments_path):
    with open(arguments_path, 'r') as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)


def find_files(folder_path):
    pt_files = glob.glob(os.path.join(folder_path, '*.pt'))
    json_files = glob.glob(os.path.join(folder_path, '*.json'))

    if len(pt_files) == 0:
        raise FileNotFoundError(f"No .pt file found in the folder: {folder_path}")
    elif len(pt_files) > 1:
        raise ValueError(f"Multiple .pt files found in the folder: {folder_path}. Please ensure only one exists.")

    if len(json_files) == 0:
        raise FileNotFoundError(f"No .json file found in the folder: {folder_path}")
    elif len(json_files) > 1:
        raise ValueError(f"Multiple .json files found in the folder: {folder_path}. Please ensure only one exists.")

    return pt_files[0], json_files[0]


def get_smc_grid(quick: bool = False):
    if quick:
        return [
            dict(
                n_checkpoints=4,
                tau_start=1.5,
                tau_end=0.2,
                finish_tau=0.02,
                ess_threshold=0.4,
                elite_frac=0.10,
                use_amp=True,
            ),
            dict(
                n_checkpoints=8,
                tau_start=2.0,
                tau_end=0.1,
                finish_tau=0.02,
                ess_threshold=0.6,
                elite_frac=0.10,
                use_amp=True,
            ),
        ]

    grid = []
    for n_checkpoints in [4, 8, 12]:
        for tau_start in [1.5, 2.0]:
            for tau_end in [0.1, 0.2]:
                for ess_threshold in [0.4, 0.6]:
                    cfg = dict(
                        n_checkpoints=n_checkpoints,
                        tau_start=tau_start,
                        tau_end=tau_end,
                        finish_tau=0.02,
                        ess_threshold=ess_threshold,
                        elite_frac=0.10,
                        use_amp=True,
                    )
                    grid.append(cfg)
    return grid


def get_cem_grid(quick: bool = False):
    if quick:
        return [
            dict(
                rounds=2,
                elite_frac=0.10,
                momentum=0.7,
                gumbel_map=True,
                use_top_p=True,
                use_amp=True,
                tau0=1.1,
                tau_decay=0.99,
                tau_min=0.45,
                p0=0.92,
                p_decay=0.995,
                p_min=0.75,
                smooth_eps=0.001,
            ),
            dict(
                rounds=3,
                elite_frac=0.15,
                momentum=0.85,
                gumbel_map=True,
                use_top_p=True,
                use_amp=True,
                tau0=1.1,
                tau_decay=0.99,
                tau_min=0.45,
                p0=0.92,
                p_decay=0.995,
                p_min=0.75,
                smooth_eps=0.001,
            ),
        ]

    grid = []
    for rounds in [2, 3]:
        for elite_frac in [0.10, 0.15]:
            for momentum in [0.7, 0.85]:
                cfg = dict(
                    rounds=rounds,
                    elite_frac=elite_frac,
                    momentum=momentum,
                    gumbel_map=True,
                    use_top_p=True,
                    use_amp=True,
                    tau0=1.1,
                    tau_decay=0.99,
                    tau_min=0.45,
                    p0=0.92,
                    p_decay=0.995,
                    p_min=0.75,
                    smooth_eps=0.001,
                )
                grid.append(cfg)
    return grid


def cfg_id(method, idx):
    return f"{method}_cfg{idx:02d}"


def get_effective_sample_multiplier(method: str, method_cfg: dict) -> int:
    """
    Multiplier between internal bs and total effective samples
    (forward calls to the policy network).
    """
    if method == "sampling":
        return 1
    if method == "smc":
        return int(method_cfg.get("n_checkpoints", 1))
    if method == "cem":
        return int(method_cfg.get("rounds", 1))
    return 1


def init_instance_json(instance_name: str, j: int, m: int):
    return {
        "instance": str(instance_name),
        "j": int(j),
        "m": int(m),
        "methods": {}
    }


def save_json_safely(data, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[JSON] Saved -> {os.path.abspath(path)}", flush=True)
    except Exception as e:
        print(f"[JSON][ERROR] Failed to save {path}: {e}", flush=True)


def update_instance_json(inst_data: dict,
                         method: str,
                         cfg_id_str: str,
                         cfg: dict,
                         budget: int,
                         stats_rep: dict):
    """
    Update per-instance JSON structure with best-gap arrays and timing info for this
    (method, cfg_id, budget).
    """
    methods = inst_data.setdefault("methods", {})
    mentry = methods.setdefault(method, {"configs": {}})
    cfgs = mentry.setdefault("configs", {})

    if cfg is None:
        cfg = {}
    cfg_clean = {}
    for k, v in cfg.items():
        if isinstance(v, (np.generic,)):
            v = v.item()
        cfg_clean[str(k)] = v

    centry = cfgs.setdefault(cfg_id_str, {
        "hyperparams": cfg_clean,
        "budgets": {}
    })
    budgets_dict = centry.setdefault("budgets", {})

    best_gaps = [float(x) for x in stats_rep["best_gap_per_rep"]]
    best_ms = [float(x) for x in stats_rep["best_ms_per_rep"]]
    times_per_rep = [float(x) for x in stats_rep.get("times_per_rep", [])]

    budgets_dict[str(int(budget))] = {
        # Budget info
        "beta": int(stats_rep.get("budget_total", budget)),
        "n_samples_effective": int(stats_rep["n_samples_effective"]),
        "bs_internal": int(stats_rep["bs_internal"]),
        "sample_multiplier": int(stats_rep["sample_multiplier"]),
        "n_reps": int(stats_rep["n_reps"]),

        # Raw per-repetition best values
        "best_gaps": best_gaps,
        "best_makespans": best_ms,

        # Timing info
        "time_mean": float(stats_rep["time_mean"]),
        "time_std": float(stats_rep["time_std"]),
        "times_per_rep": times_per_rep,
    }


# ---------------------------------------------------------
# One stochastic run (one repetition)
# ---------------------------------------------------------

@torch.no_grad()
def eval_single_instance_once(
    ins,
    encoder,
    decoder,
    method: str,
    bs_internal: int,
    method_cfg: dict,
    device: str = "cuda",
):
    encoder.eval()
    decoder.eval()

    start = time.time()

    if method == "sampling":
        trajs, ptrs, mss = sampling(ins, encoder, decoder, bs=bs_internal, device=device)
    elif method == "smc":
        trajs, ptrs, mss = _smc_sampling_fast_improved(
            ins,
            encoder,
            decoder,
            bs=bs_internal,
            device=device,
            **method_cfg,
        )
    elif method == "cem":
        trajs, ptrs, mss = _cem_guided_sampling(
            ins,
            encoder,
            decoder,
            bs=bs_internal,
            device=device,
            **method_cfg,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    time_elapsed = time.time() - start

    mss_np = mss.detach().cpu().numpy().astype(float)

    # Reference makespan (assumed optimal)
    makespan_opt = ins["makespan"]
    if torch.is_tensor(makespan_opt):
        makespan_opt = makespan_opt.item()
    makespan_opt = float(makespan_opt)

    # gap = (makespan - reference) / reference * 100
    gaps_all = (mss_np / makespan_opt - 1.0) * 100.0

    ms_min = float(mss_np.min())
    ms_max = float(mss_np.max())
    ms_mean = float(mss_np.mean())

    gap_min = float(gaps_all.min())
    gap_max = float(gaps_all.max())
    gap_mean = float(gaps_all.mean())

    return {
        "gap_min": gap_min,
        "gap_mean": gap_mean,
        "gap_max": gap_max,
        "ms_min": ms_min,
        "ms_mean": ms_mean,
        "ms_max": ms_max,
        "time_elapsed": float(time_elapsed),
    }


# ---------------------------------------------------------
# n_reps repetitions for one (instance, method, budget, cfg)
# ---------------------------------------------------------

def eval_single_instance_repeated(
    ins,
    encoder,
    decoder,
    method: str,
    budget: int,
    method_cfg: dict,
    device: str = "cuda",
    base_seed: int = 42,
    n_reps: int = 50,
):
    """
    n_reps repetitions.
    Fairness rule:
      sample_multiplier = 1 (sampling), n_checkpoints (smc), rounds (cem)
      bs_internal = floor(beta / sample_multiplier)
      effective_samples = bs_internal * sample_multiplier
    We record per-repetition best gaps/makespans and timing.
    """

    mult = get_effective_sample_multiplier(method, method_cfg)
    mult = max(mult, 1)
    bs_internal = max(1, budget // mult)
    effective_samples = bs_internal * mult

    min_gap = np.zeros(n_reps, dtype=float)
    avg_gap = np.zeros(n_reps, dtype=float)
    max_gap = np.zeros(n_reps, dtype=float)

    min_ms = np.zeros(n_reps, dtype=float)
    avg_ms = np.zeros(n_reps, dtype=float)
    max_ms = np.zeros(n_reps, dtype=float)

    times = np.zeros(n_reps, dtype=float)

    desc = f"{method}, β={budget}, bs={bs_internal}, mult={mult}"
    for r in tqdm(range(n_reps), desc=desc, leave=False):
        seed_r = base_seed + r

        torch.manual_seed(seed_r)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_r)
        np.random.seed(seed_r)
        random.seed(seed_r)

        stats = eval_single_instance_once(
            ins=ins,
            encoder=encoder,
            decoder=decoder,
            method=method,
            bs_internal=bs_internal,
            method_cfg=method_cfg,
            device=device,
        )

        min_gap[r] = stats["gap_min"]
        avg_gap[r] = stats["gap_mean"]
        max_gap[r] = stats["gap_max"]

        min_ms[r] = stats["ms_min"]
        avg_ms[r] = stats["ms_mean"]
        max_ms[r] = stats["ms_max"]

        times[r] = stats["time_elapsed"]

    def summarize(arr):
        if len(arr) > 1:
            return float(arr.mean()), float(arr.std(ddof=1))
        else:
            return float(arr.mean()), 0.0

    mg_mean, mg_std = summarize(min_gap)
    ag_mean, ag_std = summarize(avg_gap)
    xg_mean, xg_std = summarize(max_gap)

    mms_mean, mms_std = summarize(min_ms)
    ams_mean, ams_std = summarize(avg_ms)
    xms_mean, xms_std = summarize(max_ms)
    t_mean, t_std = summarize(times)

    best_gap_min_over_reps = float(min_gap.min())
    best_gap_max_over_reps = float(min_gap.max())
    best_gap_avg_over_reps = float(min_gap.mean())

    best_ms_min_over_reps = float(min_ms.min())
    best_ms_max_over_reps = float(min_ms.max())
    best_ms_avg_over_reps = float(min_ms.mean())

    return {
        "gap_min_mean": mg_mean,
        "gap_min_std": mg_std,
        "gap_avg_mean": ag_mean,
        "gap_avg_std": ag_std,
        "gap_max_mean": xg_mean,
        "gap_max_std": xg_std,

        "ms_min_mean": mms_mean,
        "ms_min_std": mms_std,
        "ms_avg_mean": ams_mean,
        "ms_avg_std": ams_std,
        "ms_max_mean": xms_mean,
        "ms_max_std": xms_std,

        "time_mean": t_mean,
        "time_std": t_std,

        "n_reps": int(n_reps),

        "budget_total": int(budget),
        "n_samples_effective": int(effective_samples),
        "bs_internal": int(bs_internal),
        "sample_multiplier": int(mult),

        "best_gap_min_over_reps": best_gap_min_over_reps,
        "best_gap_max_over_reps": best_gap_max_over_reps,
        "best_gap_avg_over_reps": best_gap_avg_over_reps,

        "best_ms_min_over_reps": best_ms_min_over_reps,
        "best_ms_max_over_reps": best_ms_max_over_reps,
        "best_ms_avg_over_reps": best_ms_avg_over_reps,

        "best_gap_per_rep": min_gap.tolist(),
        "best_ms_per_rep": min_ms.tolist(),
        "times_per_rep": times.tolist(),
    }


# ---------------------------------------------------------
# Global summary across instances (with repetition variability)
# ---------------------------------------------------------

def compute_global_summary(all_data):
    """
    For each (beta, method, config) we have per-instance arrays over repetitions:

      g[i, r] = best_gaps for instance i, repetition r
      t[i, r] = times for instance i, repetition r

    For each instance i:

      # gaps
      inst_min_gap_i   = min_r g[i, r]
      inst_max_gap_i   = max_r g[i, r]
      inst_mean_gap_i  = mean_r g[i, r]
      inst_std_gap_i   = std_r  g[i, r] (0 if n_reps == 1)

      # times
      inst_mean_time_i = mean_r t[i, r]
      inst_std_time_i  = std_r  t[i, r] (0 if n_reps == 1)

    Across instances (k instances):

      # min / max use variability across instances
      min_gap_mean = mean_i(inst_min_gap_i)
      min_gap_std  = std_i(inst_min_gap_i)
      max_gap_mean = mean_i(inst_max_gap_i)
      max_gap_std  = std_i(inst_max_gap_i)

      # average gap:
      avg_gap_mean = mean_i(inst_mean_gap_i)
      avg_gap_std  = RMS_i(inst_std_gap_i)
        where RMS(...) = sqrt(mean_i(inst_std_gap_i^2))

      # time:
      time_mean = mean_i(inst_mean_time_i)
      time_std  = RMS_i(inst_std_time_i)

    So:
      - min/max shading appears only with multiple instances.
      - avg-gap and time shading reflect repetition variability and
        are non-zero even with a single instance (if n_reps > 1).
    """
    instances = all_data.get("instances", {})
    if not instances:
        return {}

    budgets = all_data.get("_meta", {}).get("budgets", [])
    summary = {"per_budget": {}}

    for beta in budgets:
        beta_str = str(int(beta))
        beta_entry = {}  # method -> cfg_id -> aggregated over instances

        for inst_name, inst_data in instances.items():
            methods = inst_data.get("methods", {})
            for method, mentry in methods.items():
                cfgs = mentry.get("configs", {})
                for cfg_id_str, centry in cfgs.items():
                    budget_data = centry.get("budgets", {}).get(beta_str)
                    if not budget_data:
                        continue

                    best_gaps = budget_data.get("best_gaps", [])
                    if not best_gaps:
                        continue
                    best_gaps_arr = np.asarray(best_gaps, dtype=float)

                    # Per-instance gap stats across repetitions
                    inst_min_gap = float(best_gaps_arr.min())
                    inst_max_gap = float(best_gaps_arr.max())
                    inst_mean_gap = float(best_gaps_arr.mean())
                    if best_gaps_arr.size > 1:
                        inst_std_gap = float(best_gaps_arr.std(ddof=1))
                    else:
                        inst_std_gap = 0.0

                    times_per_rep = budget_data.get("times_per_rep", [])
                    if times_per_rep:
                        times_arr = np.asarray(times_per_rep, dtype=float)
                        inst_mean_time = float(times_arr.mean())
                        if times_arr.size > 1:
                            inst_std_time = float(times_arr.std(ddof=1))
                        else:
                            inst_std_time = 0.0
                    else:
                        inst_mean_time = None
                        inst_std_time = None

                    meth_dict = beta_entry.setdefault(method, {})
                    cfg_dict = meth_dict.setdefault(cfg_id_str, {
                        "gap_min_per_instance": [],
                        "gap_max_per_instance": [],
                        "gap_mean_per_instance": [],
                        "gap_std_per_instance": [],
                        "time_mean_per_instance": [],
                        "time_std_per_instance": [],
                    })

                    cfg_dict["gap_min_per_instance"].append(inst_min_gap)
                    cfg_dict["gap_max_per_instance"].append(inst_max_gap)
                    cfg_dict["gap_mean_per_instance"].append(inst_mean_gap)
                    cfg_dict["gap_std_per_instance"].append(inst_std_gap)

                    if inst_mean_time is not None:
                        cfg_dict["time_mean_per_instance"].append(inst_mean_time)
                        cfg_dict["time_std_per_instance"].append(inst_std_time)

        # Aggregate across instances
        for method, cfgs in beta_entry.items():
            for cfg_id_str, vals in list(cfgs.items()):

                def mean_std(lst):
                    arr = np.asarray(lst, dtype=float)
                    if arr.size == 0:
                        return None, None
                    if arr.size == 1:
                        return float(arr[0]), 0.0
                    return float(arr.mean()), float(arr.std(ddof=1))

                # std across instances of per-instance min/max (as per spec)
                min_gap_mean, min_gap_std = mean_std(vals["gap_min_per_instance"])
                max_gap_mean, max_gap_std = mean_std(vals["gap_max_per_instance"])

                # avg gap: mean of per-instance means
                gap_mean_arr = np.asarray(vals["gap_mean_per_instance"], dtype=float)
                if gap_mean_arr.size > 0:
                    avg_gap_mean = float(gap_mean_arr.mean())
                else:
                    avg_gap_mean = None

                # avg gap std: RMS of per-instance std across repetitions
                gap_std_arr = np.asarray(vals["gap_std_per_instance"], dtype=float)
                if gap_std_arr.size > 0:
                    avg_gap_std = float(np.sqrt(np.mean(gap_std_arr ** 2)))
                else:
                    avg_gap_std = None

                # time: mean of per-instance means, std = RMS of per-instance stds
                time_mean_arr = np.asarray(vals["time_mean_per_instance"], dtype=float)
                time_std_arr = np.asarray(vals["time_std_per_instance"], dtype=float)

                if time_mean_arr.size > 0:
                    time_mean = float(time_mean_arr.mean())
                else:
                    time_mean = None

                if time_std_arr.size > 0:
                    time_std = float(np.sqrt(np.mean(time_std_arr ** 2)))
                else:
                    time_std = None

                n_instances = len(vals["gap_min_per_instance"])

                cfg_summary = {
                    "n_instances": int(n_instances),

                    "min_gap_mean": min_gap_mean,
                    "min_gap_std": min_gap_std,
                    "max_gap_mean": max_gap_mean,
                    "max_gap_std": max_gap_std,
                    "avg_gap_mean": avg_gap_mean,
                    "avg_gap_std": avg_gap_std,

                    "time_mean": time_mean,
                    "time_std": time_std,
                }
                cfgs[cfg_id_str] = cfg_summary

        summary["per_budget"][beta_str] = beta_entry

    return summary


# ---------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------

def plot_walltime_from_summary(all_data, args):
    """
    Average wall-time plot comparing methods.
    For each method and budget, we use the config that has the best avg_gap_mean
    (i.e., best hyperparameter run) and plot its wall-time stats
    with a shaded std band (std driven by repetition variability).
    """
    summary = all_data.get("_summary")
    if not summary:
        return

    per_budget = summary.get("per_budget", {})
    if not per_budget:
        print("[PLOT][WALLTIME] No data to plot yet.")
        return

    budgets = sorted(int(b) for b in per_budget.keys())

    fig, ax = plt.subplots()

    for method in METHOD_DISPLAY.keys():
        xs = []
        ys = []
        yerr = []

        for beta in budgets:
            beta_str = str(beta)
            meth_cfgs = per_budget.get(beta_str, {}).get(method, {})
            if not meth_cfgs:
                continue

            best_avg_gap = None
            chosen_time_mean = None
            chosen_time_std = None

            for cfg_id_str, stats in meth_cfgs.items():
                if stats.get("n_instances", 0) == 0:
                    continue
                tmean = stats.get("time_mean", None)
                avg_gap = stats.get("avg_gap_mean", None)
                tstd = stats.get("time_std", None)
                if tmean is None or avg_gap is None or tstd is None:
                    continue
                # choose config with best avg gap, then look at its time
                if (best_avg_gap is None) or (avg_gap < best_avg_gap):
                    best_avg_gap = avg_gap
                    chosen_time_mean = tmean
                    chosen_time_std = tstd

            if chosen_time_mean is None:
                continue

            xs.append(beta)
            ys.append(chosen_time_mean)
            yerr.append(chosen_time_std if chosen_time_std is not None else 0.0)

        if xs:
            xs = np.array(xs, dtype=float)
            ys = np.array(ys, dtype=float)
            yerr = np.array(yerr, dtype=float)
            yerr = np.maximum(yerr, 0.0)

            color = METHOD_COLORS.get(method, None)

            # Mean line
            ax.plot(
                xs,
                ys,
                marker="o",
                linestyle="-",
                label=METHOD_DISPLAY.get(method, method),
                color=color,
            )

            # Shaded std band
            lower = ys - yerr
            upper = ys + yerr
            ax.fill_between(
                xs,
                lower,
                upper,
                color=color,
                alpha=0.2,
            )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Budget β (effective forward calls)")
    ax.set_ylabel("Average wall time per repetition (s)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig_path = os.path.join(args.output_dir, "summary_walltime.pdf")
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"[PLOT][WALLTIME] Saved {fig_path}", flush=True)


def plot_hyperparam_comparisons(all_data, args):
    """
    For each method, plot hyperparameter comparison:
    - x-axis: β
    - y-axis: min_gap_mean / avg_gap_mean / max_gap_mean
    - one curve per config
    Saved under output_dir/hyperparam_plots/<method>/...
    """
    summary = all_data.get("_summary")
    if not summary:
        return

    per_budget = summary.get("per_budget", {})
    if not per_budget:
        print("[PLOT][HYPER] No data to plot yet.")
        return

    out_root = os.path.join(args.output_dir, "hyperparam_plots")
    os.makedirs(out_root, exist_ok=True)

    budgets_sorted = sorted(int(b) for b in per_budget.keys())

    metrics = [
        ("min_gap_mean", "Best-gap per instance: min over reps (%)", "min_gap"),
        ("avg_gap_mean", "Best-gap per instance: avg over reps (%)", "avg_gap"),
        ("max_gap_mean", "Best-gap per instance: max over reps (%)", "max_gap"),
    ]

    for method in METHOD_DISPLAY.keys():
        method_dir = os.path.join(out_root, method)
        os.makedirs(method_dir, exist_ok=True)

        # Collect all config ids seen for this method
        all_cfg_ids = set()
        for beta_str, meths in per_budget.items():
            meth_cfgs = meths.get(method, {})
            all_cfg_ids.update(meth_cfgs.keys())

        if not all_cfg_ids:
            continue

        for metric_key, ylabel, metric_slug in metrics:
            fig, ax = plt.subplots()

            for cfg_id_str in sorted(all_cfg_ids):
                xs = []
                ys = []
                for beta in budgets_sorted:
                    beta_str = str(beta)
                    stats = per_budget.get(beta_str, {}).get(method, {}).get(cfg_id_str)
                    if not stats or stats.get("n_instances", 0) == 0:
                        continue
                    mval = stats.get(metric_key, None)
                    if mval is None:
                        continue
                    xs.append(beta)
                    ys.append(mval)
                if xs:
                    xs = np.array(xs, dtype=float)
                    ys = np.array(ys, dtype=float)
                    ax.plot(xs, ys, marker="o", linestyle="-", label=cfg_id_str)

            if not ax.lines:
                plt.close(fig)
                continue

            ax.set_xscale("log", base=2)
            ax.set_xlabel("Budget β (effective forward calls)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{METHOD_DISPLAY.get(method, method)} – hyperparameter comparison")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            ax.legend(fontsize=BASE_FONTSIZE * SCALE * 0.7, ncol=2)

            fig_path = os.path.join(method_dir, f"{method}_hyper_{metric_slug}.pdf")
            plt.savefig(fig_path)
            plt.close(fig)
            print(f"[PLOT][HYPER] Saved {fig_path}", flush=True)


def plot_gap_vs_problem_size(all_data, args):
    """
    For each method, plot:
      y-axis: average of instance-level best gap
      x-axis: explicit problem size "m x j" (categorical)
      Different colors: different budgets β

    Instance-level best gap for (method, β) = min over configs of:
      min(best_gaps_per_rep for that config and instance, β)
    """
    instances = all_data.get("instances", {})
    if not instances:
        print("[PLOT][SIZE] No instances to plot.")
        return

    meta = all_data.get("_meta", {})
    budgets = meta.get("budgets", [])
    if not budgets:
        print("[PLOT][SIZE] No budgets in meta.")
        return

    methods = meta.get("methods", list(METHOD_DISPLAY.keys()))

    out_dir = os.path.join(args.output_dir, "size_plots")
    os.makedirs(out_dir, exist_ok=True)

    for method in methods:
        # For each budget: map (m, j) -> list of best gaps
        per_budget_size_to_gaps = {int(b): {} for b in budgets}
        size_pairs_set = set()  # to know all (m, j) we ever see for this method

        for inst_name, inst_data in instances.items():
            methods_dict = inst_data.get("methods", {})
            mentry = methods_dict.get(method)
            if not mentry:
                continue

            j = int(inst_data.get("j", 0))
            m = int(inst_data.get("m", 0))
            size_pair = (m, j)
            size_pairs_set.add(size_pair)

            cfgs = mentry.get("configs", {})

            for beta in budgets:
                beta_str = str(int(beta))
                best_gap_for_inst_beta = None

                for cfg_id_str, centry in cfgs.items():
                    bdata = centry.get("budgets", {}).get(beta_str)
                    if not bdata:
                        continue
                    best_gaps = bdata.get("best_gaps", [])
                    if not best_gaps:
                        continue
                    inst_best_this_cfg = float(np.min(best_gaps))
                    if (best_gap_for_inst_beta is None) or (inst_best_this_cfg < best_gap_for_inst_beta):
                        best_gap_for_inst_beta = inst_best_this_cfg

                if best_gap_for_inst_beta is None:
                    continue

                size_dict = per_budget_size_to_gaps[int(beta)]
                size_list = size_dict.setdefault(size_pair, [])
                size_list.append(best_gap_for_inst_beta)

        if not size_pairs_set:
            # No data at all for this method
            continue

        # Sort size pairs in a reasonable order: by m*j, then m, then j
        size_pairs_sorted = sorted(
            size_pairs_set,
            key=lambda p: (p[0] * p[1], p[0], p[1])
        )
        # Map each (m, j) to an x-index
        size_to_idx = {sz: idx for idx, sz in enumerate(size_pairs_sorted)}
        x_positions = np.arange(len(size_pairs_sorted))
        x_labels = [f"{m}x{j}" for (m, j) in size_pairs_sorted]

        fig, ax = plt.subplots()
        any_data = False

        # One curve per budget
        for beta in budgets:
            size_to_gaps = per_budget_size_to_gaps.get(int(beta), {})
            if not size_to_gaps:
                continue

            xs = []
            ys = []

            for size_pair in size_pairs_sorted:
                gaps_list = size_to_gaps.get(size_pair)
                if not gaps_list:
                    continue
                xs.append(size_to_idx[size_pair])
                gaps_arr = np.asarray(gaps_list, dtype=float)
                ys.append(float(gaps_arr.mean()))

            if not xs:
                continue

            xs = np.array(xs, dtype=float)
            ys = np.array(ys, dtype=float)

            ax.plot(xs, ys, marker="o", linestyle="-", label=f"β={beta}")
            any_data = True

        if not any_data:
            plt.close(fig)
            continue

        ax.set_xlabel("Problem size (machines × jobs)")
        ax.set_ylabel("Average best gap per instance (%)")
        ax.set_title(f"{METHOD_DISPLAY.get(method, method)} – gap vs problem size")

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

        fig_path = os.path.join(out_dir, f"{method}_gap_vs_size.pdf")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT][SIZE] Saved {fig_path}", flush=True)


# ---------------------------------------------------------
# Plotting from JSON summary (main global plots)
# ---------------------------------------------------------

def plot_from_summary(all_data, args):
    """
    Create/update plots from all_data["_summary"] and per-instance data.
    - Global gap plots (min/avg/max) using best config per method & budget,
      with shaded std (min/max: cross-instance; avg: repetition-driven)
    - Average wall-time plot (with shaded std driven by repetition variability)
    - Hyperparameter comparison plots per method
    - Gap vs problem size plots per method
    """
    summary = all_data.get("_summary")
    if not summary:
        summary = compute_global_summary(all_data)
        all_data["_summary"] = summary

    per_budget = summary.get("per_budget", {})
    if not per_budget:
        print("[PLOT] No data to plot yet.")
        return

    budgets = sorted(int(b) for b in per_budget.keys())

    # 1) Global gap plots: for each metric, best config per method & budget
    metrics = [
        ("min_gap", "Best min gap across reps (%)", "min_gap_mean", "min_gap_std"),
        ("avg_gap", "Average gap across reps (%)", "avg_gap_mean", "avg_gap_std"),
        ("max_gap", "Worst min gap across reps (%)", "max_gap_mean", "max_gap_std"),
    ]

    for metric_slug, ylabel, mean_key, std_key in metrics:
        fig, ax = plt.subplots()

        for method in METHOD_DISPLAY.keys():
            xs = []
            ys = []
            yerr = []

            for beta in budgets:
                beta_str = str(beta)
                meth_cfgs = per_budget.get(beta_str, {}).get(method, {})
                if not meth_cfgs:
                    continue

                best_mean = None
                best_std = None

                for cfg_id_str, stats in meth_cfgs.items():
                    if stats.get("n_instances", 0) == 0:
                        continue
                    mval = stats.get(mean_key, None)
                    sval = stats.get(std_key, None)
                    if mval is None or sval is None:
                        continue
                    if best_mean is None or mval < best_mean:
                        best_mean = mval
                        best_std = sval

                if best_mean is None:
                    continue

                xs.append(beta)
                ys.append(best_mean)
                yerr.append(best_std if best_std is not None else 0.0)

            if xs:
                xs = np.array(xs, dtype=float)
                ys = np.array(ys, dtype=float)
                yerr = np.array(yerr, dtype=float)

                # Ensure non-negative std
                yerr = np.maximum(yerr, 0.0)

                color = METHOD_COLORS.get(method, None)

                # Mean line
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linestyle="-",
                    label=METHOD_DISPLAY.get(method, method),
                    color=color,
                )

                # Shaded std band (mean ± std)
                lower = ys - yerr
                upper = ys + yerr
                ax.fill_between(
                    xs,
                    lower,
                    upper,
                    color=color,
                    alpha=0.2,
                )

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Budget β (effective forward calls)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

        fig_path = os.path.join(args.output_dir, f"summary_{metric_slug}.pdf")
        plt.savefig(fig_path)
        plt.close(fig)
        print(f"[PLOT] Saved {fig_path}", flush=True)

    # 2) Average wall-time plot
    plot_walltime_from_summary(all_data, args)

    # 3) Hyperparameter comparison plots per method
    plot_hyperparam_comparisons(all_data, args)

    # 4) Gap vs problem size plots per method
    plot_gap_vs_problem_size(all_data, args)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="JSSP: sampling vs SMC vs CEM with fair budget, JSON logging, and live plots."
    )

    parser.add_argument(
        "-folder_path",
        type=str,
        required=True,
        help="Path to folder containing model checkpoint (.pt) and arguments (.json)",
    )
    parser.add_argument(
        "-benchmark",
        type=str,
        default="TA",
        help="Benchmark name.",
    )
    parser.add_argument(
        "-seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "-num_instances",
        type=int,
        default=None,
        help="Number of instances to test (default: all).",
    )
    parser.add_argument(
        "-max_jobs",
        type=int,
        default=None,
        help="Maximum number of jobs (filter instances).",
    )
    parser.add_argument(
        "-max_machines",
        type=int,
        default=None,
        help="Maximum number of machines (filter instances).",
    )
    parser.add_argument(
        "-random",
        action="store_true",
        help="Randomly shuffle instances.",
    )
    parser.add_argument(
        "-output_dir",
        type=str,
        default="hyper_smc_cem_50reps",
        help="Directory to store JSON and plots.",
    )
    parser.add_argument(
        "-infer_sch_expert",
        action="store_true",
        help="Use SchedulExpert architecture.",
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Use tiny hyperparameter grids for SMC and CEM.",
    )
    parser.add_argument(
        "-n_reps",
        type=int,
        default=3,
        help="Number of repetitions per config.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {os.path.abspath(args.output_dir)}", flush=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Quick-test mode: {args.quick_test}")
    print(f"Repetitions per config: {args.n_reps}")

    if args.infer_sch_expert:
        from architectures.SchedulExpert import GATEncoder, MHADecoder
    else:
        raise RuntimeError("This script currently assumes SchedulExpert. Use -infer_sch_expert.")

    model_path, arguments_path = find_files(args.folder_path)
    print(f"Detected model checkpoint: {model_path}")
    print(f"Detected arguments file: {arguments_path}")

    args_loaded = load_args(arguments_path)

    encoder = GATEncoder(
        input_size=15,
        hidden_size=args_loaded.enc_hidden,
        embed_size=args_loaded.enc_out,
        n_experts=args_loaded.n_experts,
    ).to(device)

    decoder = MHADecoder(
        encoder_size=encoder.out_size,
        context_size=JobShopStates.size,
        hidden_size=args_loaded.mem_hidden,
        mem_size=args_loaded.mem_out,
        clf_size=args_loaded.clf_hidden,
    ).to(device)

    print("Loading model weights...")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
    elif isinstance(checkpoint, tuple) and len(checkpoint) == 2:
        encoder.load_state_dict(checkpoint[0])
        decoder.load_state_dict(checkpoint[1])
    else:
        raise ValueError(
            "Invalid checkpoint format. Expected dict({'encoder','decoder'}) or "
            "tuple(encoder_state, decoder_state)."
        )
    print("Model loaded.")

    if args.benchmark == "train":
        path = "dataset5k/"
    else:
        path = f"./benchmarks/{args.benchmark}"

    all_files = [
        f for f in os.listdir(path)
        if not (f.startswith(".") or f.startswith("cached"))
    ]
    if not all_files:
        raise RuntimeError(f"No instance files found in {path}")

    if args.random:
        rng = np.random.RandomState(args.seed)
        rng.shuffle(all_files)

    if args.num_instances is not None:
        all_files = all_files[:args.num_instances]

    print(f"Evaluating on {len(all_files)} instances from {path}")

    budgets = [16, 32, 64, 128, 256, 512, 1024]
    smc_grid = get_smc_grid(quick=args.quick_test)
    cem_grid = get_cem_grid(quick=args.quick_test)

    print(f"SMC grid size: {len(smc_grid)} (quick_test={args.quick_test})")
    print(f"CEM grid size: {len(cem_grid)} (quick_test={args.quick_test})")

    json_dir = os.path.join(args.output_dir, "per_instance_json")
    os.makedirs(json_dir, exist_ok=True)

    all_json_path = os.path.join(args.output_dir, "all_instances.json")

    # Initialize or load global JSON
    if os.path.exists(all_json_path):
        try:
            with open(all_json_path, "r") as f:
                all_data = json.load(f)
        except Exception:
            all_data = {}
    else:
        all_data = {}

    all_data.setdefault("_meta", {})
    all_data["_meta"]["n_reps"] = int(args.n_reps)
    all_data["_meta"]["budgets"] = budgets
    all_data["_meta"]["methods"] = list(METHOD_DISPLAY.keys())
    all_data["_meta"]["gap_definition"] = (
        "gap = (makespan - reference) / reference * 100; "
        "reference = instance['makespan'] (assumed optimal)."
    )
    all_data["_meta"]["beta_unit"] = "one forward call to the policy network."
    all_data["_meta"]["rounding_rule"] = (
        "For each method, sample_multiplier = 1 (sampling), "
        "n_checkpoints (smc), or rounds (cem). "
        "bs_internal = floor(beta / sample_multiplier). "
        "Effective samples = bs_internal * sample_multiplier <= beta."
    )
    all_data.setdefault("instances", {})

    # Main loop: instances / methods / configs / budgets
    for inst_idx, file in enumerate(all_files):
        instance_path = os.path.join(path, file)
        instance = load_data(instance_path, device=device)

        j = int(instance["j"])
        m = int(instance["m"])
        if args.max_jobs is not None and j > args.max_jobs:
            print(f"Skipping {file}: j={j} > max_jobs={args.max_jobs}")
            continue
        if args.max_machines is not None and m > args.max_machines:
            print(f"Skipping {file}: m={m} > max_machines={args.max_machines}")
            continue

        print(
            f"\n===== Instance [{inst_idx+1}/{len(all_files)}]: {file} (j={j}, m={m}) =====",
            flush=True,
        )

        # Load or init instance JSON inside all_data
        inst_json = all_data["instances"].get(file)
        if inst_json is None:
            inst_json = init_instance_json(file, j, m)

        # -------- Ordinary sampling --------
        for budget in budgets:
            print(f"  [Sampling] β={budget}, {args.n_reps} reps", flush=True)

            base_seed = (
                args.seed
                + 100000 * inst_idx
                + 1000 * budgets.index(budget)
                + 0
            )

            stats_rep = eval_single_instance_repeated(
                ins=instance,
                encoder=encoder,
                decoder=decoder,
                method="sampling",
                budget=budget,
                method_cfg={},
                device=device,
                base_seed=base_seed,
                n_reps=args.n_reps,
            )

            update_instance_json(
                inst_json,
                method="sampling",
                cfg_id_str="sampling",
                cfg={},
                budget=budget,
                stats_rep=stats_rep,
            )

            # Immediately sync into global JSON and plots
            all_data["instances"][file] = inst_json
            all_data["_summary"] = compute_global_summary(all_data)
            save_json_safely(all_data, all_json_path)
            plot_from_summary(all_data, args)

        # -------- SMC --------
        for i, cfg in enumerate(smc_grid):
            cfg_name = cfg_id("smc", i)
            print(f"\n[SMC] Config {cfg_name}: {cfg}", flush=True)
            for budget in budgets:
                print(f"  [SMC] β={budget}, {args.n_reps} reps", flush=True)

                base_seed = (
                    args.seed
                    + 100000 * inst_idx
                    + 1000 * budgets.index(budget)
                    + 10 * (i + 1)
                )

                stats_rep = eval_single_instance_repeated(
                    ins=instance,
                    encoder=encoder,
                    decoder=decoder,
                    method="smc",
                    budget=budget,
                    method_cfg=cfg,
                    device=device,
                    base_seed=base_seed,
                    n_reps=args.n_reps,
                )

                update_instance_json(
                    inst_json,
                    method="smc",
                    cfg_id_str=cfg_name,
                    cfg=cfg,
                    budget=budget,
                    stats_rep=stats_rep,
                )

                all_data["instances"][file] = inst_json
                all_data["_summary"] = compute_global_summary(all_data)
                save_json_safely(all_data, all_json_path)
                plot_from_summary(all_data, args)

        # -------- CEM --------
        for i, cfg in enumerate(cem_grid):
            cfg_name = cfg_id("cem", i)
            print(f"\n[CEM] Config {cfg_name}: {cfg}", flush=True)
            for budget in budgets:
                print(f"  [CEM] β={budget}, {args.n_reps} reps", flush=True)

                base_seed = (
                    args.seed
                    + 100000 * inst_idx
                    + 1000 * budgets.index(budget)
                    + 20 * (i + 1)
                )

                stats_rep = eval_single_instance_repeated(
                    ins=instance,
                    encoder=encoder,
                    decoder=decoder,
                    method="cem",
                    budget=budget,
                    method_cfg=cfg,
                    device=device,
                    base_seed=base_seed,
                    n_reps=args.n_reps,
                )

                update_instance_json(
                    inst_json,
                    method="cem",
                    cfg_id_str=cfg_name,
                    cfg=cfg,
                    budget=budget,
                    stats_rep=stats_rep,
                )

                all_data["instances"][file] = inst_json
                all_data["_summary"] = compute_global_summary(all_data)
                save_json_safely(all_data, all_json_path)
                plot_from_summary(all_data, args)

        # Also dump per-instance JSON file (nice for debugging)
        json_path = os.path.join(json_dir, f"{os.path.splitext(file)[0]}.json")
        save_json_safely(inst_json, json_path)

    print("Finished all instances.", flush=True)

    # Final summary & plots (should already be up to date, but no harm)
    all_data["_summary"] = compute_global_summary(all_data)
    save_json_safely(all_data, all_json_path)
    plot_from_summary(all_data, args)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
# python hyperparam_search_smc_cem.py   -folder_path checkpoints/SchedulExpert   -benchmark TA   -num_instances 2   -seed 42    -infer_sch_expert   -output_dir hyper_smc_cem --quick_test -n_reps 10