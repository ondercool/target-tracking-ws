#!/usr/bin/env python3
"""Compute tracking precision / RMSE tables and figures for the target lock tests.

Data layout (per scenario folder): <scenario>_NCAAUG_<k>.csv, k = 0..19.
Trials 0-9 use the inflated measurement matrix, trials 10-19 the constant one.

Outputs (under vb_controller/):
    tables/<scenario>/trials.csv   per-trial precision and RMSE values
    tables/<scenario>/stats.csv    mean / variance over trials, per model
    figures/<scenario>_timeseries.png
    figures/<scenario>_precision.png
    figures/<scenario>_rmse.png

Run:  python3 compute_metrics.py [--sample-trial 0]
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

SCENARIOS = ("linear_forward", "driveby", "sinusoidal")
MODELS = {"inflated": range(0, 10), "constant": range(10, 20)}
MODEL_LABELS = {"inflated": "Inflated R", "constant": "Constant R"}
MODEL_COLORS = {"inflated": "#4C78A8", "constant": "#E45756"}
ERR_COLORS = {"u_err": "#4C78A8", "v_err": "#F58518", "pixel_err": "#54A24B"}

# Camera / target parameters (depth and radius in cm), same as lock_target_NCA*.py
FOCAL_PX = 1481.37
REAL_RADIUS = 9.0
LASER_Y_OFFSET = 2.5
DEPTH_ERR_MAX = 100.0  # assumed worst-case depth error (cm)

DATA_DIR = Path(__file__).resolve().parent
PKG_DIR = DATA_DIR.parent
TABLES_DIR = PKG_DIR / "tables"
FIGURES_DIR = PKG_DIR / "figures"

def radius_threshold(depth):
    """r_th = f R / (Z + dZ) - dZ f y_off / (Z^2 - dZ Z); NaN where Z <= dZ."""
    z = np.asarray(depth, dtype=float)
    d = DEPTH_ERR_MAX
    with np.errstate(divide="ignore", invalid="ignore"):
        r_th = FOCAL_PX * REAL_RADIUS / (z + d) - d * FOCAL_PX * LASER_Y_OFFSET / (
            z**2 - d * z
        )
    return np.where(z > d, r_th, np.nan)


def trial_path(scenario, k):
    return DATA_DIR / scenario / f"{scenario}_NCAAUG_{k}.csv"


def load_trial(scenario, k):
    df = pd.read_csv(trial_path(scenario, k))
    df["pixel_err"] = np.hypot(df["u_err"], df["v_err"])
    return df


def trial_metrics(df):
    """Precision = rows with pixel_err < threshold / all rows. RMSE over all rows."""
    r_th = radius_threshold(df["depth"].to_numpy())
    on_target = df["pixel_err"].to_numpy() < r_th  # NaN threshold -> False
    return {
        "n_rows": len(df),
        "precision": float(on_target.sum()) / len(df),
        "u_errRMSE": float(np.sqrt(np.mean(df["u_err"] ** 2))),
        "v_errRMSE": float(np.sqrt(np.mean(df["v_err"] ** 2))),
        "pixel_errRMSE": float(np.sqrt(np.mean(df["pixel_err"] ** 2))),
        "invalid_threshold_rows": int(np.isnan(r_th).sum()),
    }


def compute_tables(scenario):
    rows = []
    for model, ks in MODELS.items():
        for i, k in enumerate(ks):
            m = trial_metrics(load_trial(scenario, k))
            rows.append({"model": model, "trial": i, "file_index": k, **m})
    trials = pd.DataFrame(rows)

    stat_cols = ["precision", "u_errRMSE", "v_errRMSE", "pixel_errRMSE"]
    stats = []
    for model in MODELS:
        sub = trials[trials["model"] == model]
        for col in stat_cols:
            stats.append(
                {
                    "model": model,
                    "metric": col,
                    "mean": sub[col].mean(),
                    "variance": sub[col].var(ddof=1),  # sample variance
                    "n_trials": len(sub),
                }
            )
    stats = pd.DataFrame(stats)

    out = TABLES_DIR / scenario
    out.mkdir(parents=True, exist_ok=True)
    trials.to_csv(out / "trials.csv", index=False)
    stats.to_csv(out / "stats.csv", index=False)
    return trials, stats


def mean_sd_bar(ax, x, mean, var, width, color, label=None):
    """Filled bar for the mean with a hollow box spanning mean +/- 1 SD."""
    sd = np.sqrt(var)
    ax.bar(x, mean, width, color=color, label=label)
    ax.add_patch(
        Rectangle(
            (x - width / 2, mean - sd),
            width,
            2 * sd,
            fill=True,
            facecolor="white",
            alpha=0.75,
            edgecolor=color,
            linewidth=1.8,
            zorder=3,
        )
    )


def stat(stats, model, metric):
    row = stats[(stats["model"] == model) & (stats["metric"] == metric)].iloc[0]
    return row["mean"], row["variance"]


def plot_precision(scenario, stats):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for i, model in enumerate(MODELS):
        mean, var = stat(stats, model, "precision")
        mean_sd_bar(ax, i, mean * 100, var * 100**2, 0.5, MODEL_COLORS[model])
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("Tracking precision (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"{scenario}: precision (mean, box = ±1 SD)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{scenario}_precision.png", dpi=200)
    plt.close(fig)


def plot_rmse(scenario, stats):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    metrics = ("u_errRMSE", "v_errRMSE", "pixel_errRMSE")
    width = 0.35
    for gi, metric in enumerate(metrics):
        for mi, model in enumerate(MODELS):
            mean, var = stat(stats, model, metric)
            mean_sd_bar(
                ax,
                gi + (mi - 0.5) * width * 1.1,
                mean,
                var,
                width,
                MODEL_COLORS[model],
                label=MODEL_LABELS[model] if gi == 0 else None,
            )
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(["u_err", "v_err", "pixel_err"])
    ax.set_ylabel("RMSE (pixels)")
    ax.set_title(f"{scenario}: RMSE (mean, box = ±1 SD over trials)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{scenario}_rmse.png", dpi=200)
    plt.close(fig)


def plot_timeseries(scenario, sample_trial):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, (model, ks) in zip(axes, MODELS.items()):
        df = load_trial(scenario, list(ks)[sample_trial])
        for col in ("u_err", "v_err", "pixel_err"):
            ax.plot(df["elapsed_time"].to_numpy(), df[col].to_numpy(), label=col, color=ERR_COLORS[col])
        ax.set_title(f"{MODEL_LABELS[model]} (trial {sample_trial})")
        ax.set_xlabel("Elapsed time (s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Error (pixels)")
    axes[0].legend(loc="upper right")
    fig.suptitle(scenario)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{scenario}_timeseries.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sample-trial",
        type=int,
        default=0,
        help="trial index (0-9) within each model used for the time-series figure",
    )
    args = ap.parse_args()
    if not 0 <= args.sample_trial <= 9:
        raise SystemExit("--sample-trial must be in 0..9")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        for k in range(20):
            if not trial_path(scenario, k).is_file():
                raise SystemExit(f"Missing file: {trial_path(scenario, k)}")
        trials, stats = compute_tables(scenario)
        if trials["invalid_threshold_rows"].sum():
            print(
                f"[warn] {scenario}: {trials['invalid_threshold_rows'].sum()} rows "
                f"with depth <= {DEPTH_ERR_MAX} cm counted as off-target"
            )
        plot_precision(scenario, stats)
        plot_rmse(scenario, stats)
        plot_timeseries(scenario, args.sample_trial)
        print(f"\n== {scenario} ==")
        print(stats.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
