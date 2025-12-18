# plot_from_csv_smc_cem.py

import argparse
import os

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl


# ---------------------------------------------------------
# Scientific plotting style
# ---------------------------------------------------------
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "figure.figsize": (5.0, 3.5),
    "legend.frameon": False,
})

METHOD_COLORS = {
    "sampling": "#0072B2",  # blue
    "smc":      "#D55E00",  # orange
    "cem":      "#009E73",  # green
}

METHOD_DISPLAY = {
    "sampling": "Ordinary sampling",
    "smc": "SMC",
    "cem": "CEM",
}


def recompute_everything(output_dir: str, benchmark: str):
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, f"raw_results_{benchmark}.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"raw_results CSV not found at: {raw_path}")

    df = pd.read_csv(raw_path)
    if df.empty:
        raise RuntimeError(f"raw_results CSV is empty: {raw_path}")

    print(f"[PLOT] Loaded {len(df)} rows from {raw_path}")
    print(f"[PLOT] Unique instances: {df['instance'].nunique()}")

    # ---------- agg_results ----------
    agg = (
        df.groupby(["method", "budget", "cfg_id"])
        .agg(
            avg_gap=("gap_best", "mean"),
            std_gap=("gap_best", "std"),
            avg_time=("time", "mean"),
            std_time=("time", "std"),
            n_instances=("instance", "nunique"),
        )
        .reset_index()
    )
    agg_path = os.path.join(output_dir, f"agg_results_{benchmark}.csv")
    agg.to_csv(agg_path, index=False)
    print(f"[PLOT] agg_results -> {os.path.abspath(agg_path)}")

    # ---------- per_instance_stats ----------
    inst_stats = (
        df.groupby(["instance", "j", "m", "method", "budget"])
        .agg(
            inst_avg_gap=("gap_best", "mean"),
            inst_std_gap=("gap_best", "std"),
            inst_avg_ms=("ms_best", "mean"),
            inst_std_ms=("ms_best", "std"),
            inst_avg_time=("time", "mean"),
            inst_std_time=("time", "std"),
            n_configs=("cfg_id", "nunique"),
        )
        .reset_index()
    )
    inst_path = os.path.join(output_dir, f"per_instance_stats_{benchmark}.csv")
    inst_stats.to_csv(inst_path, index=False)
    print(f"[PLOT] per_instance_stats -> {os.path.abspath(inst_path)}")

    if agg.empty:
        print("[PLOT] agg is empty, nothing to plot.")
        return

    # ---------- best_configs ----------
    best_rows = []
    for (method, budget), sub in agg.groupby(["method", "budget"]):
        sub = sub.dropna(subset=["avg_gap"])
        if sub.empty:
            continue
        best_idx = sub["avg_gap"].idxmin()
        best_rows.append(agg.loc[best_idx])

    if not best_rows:
        print("[PLOT] No best rows, skipping plots.")
        return

    best_df = pd.DataFrame(best_rows).sort_values(["method", "budget"])
    best_path = os.path.join(output_dir, f"best_configs_{benchmark}.csv")
    best_df.to_csv(best_path, index=False)
    print(f"[PLOT] best_configs -> {os.path.abspath(best_path)}")

    # ---------- LaTeX table ----------
    try:
        pivot = best_df.pivot(index="budget", columns="method", values="avg_gap").sort_index()
        latex_table = pivot.to_latex(
            float_format="%.3f",
            na_rep="--",
            caption=f"Average best GAP (\\%) over instances on {benchmark} benchmark.",
            label="tab:smc_cem_budget_gap",
        )
        tex_path = os.path.join(output_dir, f"gap_table_{benchmark}.tex")
        with open(tex_path, "w") as f:
            f.write(latex_table)
        print(f"[PLOT] LaTeX table -> {os.path.abspath(tex_path)}")
    except Exception as e:
        print(f"[WARN] Failed to write LaTeX table: {e}")

    # ---------- Plots ----------
    # GAP vs budget (with std shading)
    try:
        plt.figure()
        for method in sorted(best_df["method"].unique()):
            sub = best_df[best_df["method"] == method].sort_values("budget")
            x = sub["budget"].values
            y = sub["avg_gap"].values
            y_std = sub["std_gap"].fillna(0.0).values

            color = METHOD_COLORS.get(method, "#000000")
            label = METHOD_DISPLAY.get(method, method)

            plt.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                color=color,
                label=label,
            )
            plt.fill_between(
                x,
                y - y_std,
                y + y_std,
                color=color,
                alpha=0.18,
                linewidth=0.0,
            )

        plt.xscale("log", base=2)
        plt.xlabel("Budget (number of policy samples)")
        plt.ylabel("Average best GAP (%)")
        plt.title(f"GAP vs budget on {benchmark}")
        plt.legend()
        plt.tight_layout()

        gap_pdf = os.path.join(output_dir, f"gap_vs_budget_{benchmark}.pdf")
        plt.savefig(gap_pdf, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] GAP vs budget plot -> {os.path.abspath(gap_pdf)}")
    except Exception as e:
        print(f"[WARN] Failed to create GAP plot: {e}")

    # Time vs budget (with std shading)
    try:
        plt.figure()
        for method in sorted(best_df["method"].unique()):
            sub = best_df[best_df["method"] == method].sort_values("budget")
            x = sub["budget"].values
            y = sub["avg_time"].values
            y_std = sub["std_time"].fillna(0.0).values

            color = METHOD_COLORS.get(method, "#000000")
            label = METHOD_DISPLAY.get(method, method)

            plt.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                color=color,
                label=label,
            )
            plt.fill_between(
                x,
                y - y_std,
                y + y_std,
                color=color,
                alpha=0.18,
                linewidth=0.0,
            )

        plt.xscale("log", base=2)
        plt.xlabel("Budget (number of policy samples)")
        plt.ylabel("Average wall time per instance (s)")
        plt.title(f"Wall time vs budget on {benchmark}")
        plt.legend()
        plt.tight_layout()

        time_pdf = os.path.join(output_dir, f"time_vs_budget_{benchmark}.pdf")
        plt.savefig(time_pdf, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] Time vs budget plot -> {os.path.abspath(time_pdf)}")
    except Exception as e:
        print(f"[WARN] Failed to create Time plot: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild SMC/CEM plots from CSV files."
    )
    parser.add_argument(
        "-output_dir",
        type=str,
        required=True,
        help="Directory where raw_results_<benchmark>.csv is stored.",
    )
    parser.add_argument(
        "-benchmark",
        type=str,
        default="TA",
        help="Benchmark name used in file names (e.g. TA).",
    )
    args = parser.parse_args()

    recompute_everything(args.output_dir, args.benchmark)


if __name__ == "__main__":
    main()

# python plot_from_csv_smc_cem.py \
#   -output_dir hyper_smc_cem \
#   -benchmark TA