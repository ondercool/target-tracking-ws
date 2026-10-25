"""Generate scenario error time-series plots and an RMSE summary.

Run with:
    python3 /path/to/data/computePlots.py

Use --sample-trials to select one or more trial numbers for the time-series
plots. Figures are saved to data/plots by default.
"""

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

TIME_COLUMNS = ("elapsed_time",)
ERROR_COLUMNS = ("u_err", "v_err", "pixel_err")
RMSE_COLUMNS = ("u_errRMSE", "v_errRMSE", "pixel_errRMSE")
METRIC_LABELS = ("u_err", "v_err", "pixel_err")
METRIC_COLORS = ("#4C78A8", "#F58518", "#54A24B")


def read_csv_rows(path: Path, required_columns: Sequence[str]) -> List[Dict[str, str]]:
    """Read a CSV and report missing files, columns, or empty tables clearly."""
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV file not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        headers = reader.fieldnames or []
        missing = [column for column in required_columns if column not in headers]
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} contains no data rows")
    return rows


def numeric_column(
    rows: Sequence[Dict[str, str]], column: str, path: Path
) -> List[float]:
    values: List[float] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            value = float(row[column])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}:{row_number}: {column} must contain a number"
            ) from error
        if not math.isfinite(value):
            raise ValueError(f"{path}:{row_number}: {column} must be finite")
        values.append(value)
    return values


def discover_scenarios(data_dir: Path) -> List[Tuple[str, Path, Path]]:
    scenarios = []
    for scenario_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        scenario = scenario_dir.name
        stats_path = scenario_dir / f"{scenario}_stats.csv"
        if stats_path.is_file():
            scenarios.append((scenario, scenario_dir, stats_path))
    if not scenarios:
        raise FileNotFoundError(
            f"No scenario folders with matching *_stats.csv files found in {data_dir}"
        )
    return scenarios


def plot_trial(scenario: str, trial: int, scenario_dir: Path, output_dir: Path) -> None:
    csv_path = scenario_dir / f"{scenario}_{trial}.csv"
    columns = (*TIME_COLUMNS, *ERROR_COLUMNS)
    rows = read_csv_rows(csv_path, columns)
    time_values = numeric_column(rows, "elapsed_time", csv_path)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    for column in ERROR_COLUMNS:
        axis.plot(
            time_values,
            numeric_column(rows, column, csv_path),
            label=column,
            linewidth=1.5,
        )

    axis.set(
        title=f"{scenario} - trial {trial}",
        xlabel="Elapsed time (s)",
        ylabel="Error (pixels)",
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{scenario}_trial_{trial}_errors.png", dpi=160)
    plt.close(fig)


def plot_rmse_summary(
    scenarios: Sequence[Tuple[str, Path, Path]], output_path: Path
) -> None:
    scenario_names = [scenario for scenario, _, _ in scenarios]
    bar_width = 0.24
    group_positions = list(range(len(scenarios)))
    offsets = (
        -bar_width,
        0.0,
        bar_width,
    )

    fig, axis = plt.subplots(figsize=(11, 6))
    for metric_index, (metric, color, offset) in enumerate(
        zip(RMSE_COLUMNS, METRIC_COLORS, offsets)
    ):
        means: List[float] = []
        standard_deviations: List[float] = []
        positions = [position + offset for position in group_positions]

        for _, _, stats_path in scenarios:
            rows = read_csv_rows(stats_path, ("trial_no", *RMSE_COLUMNS))
            values = numeric_column(rows, metric, stats_path)
            means.append(statistics.mean(values))
            standard_deviations.append(
                statistics.stdev(values) if len(values) > 1 else 0.0
            )

        bars = axis.bar(
            positions,
            means,
            width=bar_width * 0.9,
            color=color,
            label=METRIC_LABELS[metric_index],
            zorder=2,
        )
        for bar, mean, standard_deviation in zip(bars, means, standard_deviations):
            center_x = bar.get_x() + bar.get_width() / 2
            lower = max(0.0, mean - standard_deviation)
            spread_box = Rectangle(
                (center_x - bar.get_width() * 0.27, lower),
                bar.get_width() * 0.54,
                mean + standard_deviation - lower,
                facecolor="white",
                edgecolor=color,
                linewidth=1.5,
                alpha=0.9,
                zorder=3,
            )
            axis.add_patch(spread_box)
            axis.hlines(
                mean,
                center_x - bar.get_width() * 0.27,
                center_x + bar.get_width() * 0.27,
                color=color,
                linewidth=2,
                zorder=4,
            )

    axis.set_xticks(group_positions)
    axis.set_xticklabels(scenario_names)
    axis.set_ylabel("RMSE (pixels)")
    axis.set_title("Mean RMSE by scenario and error metric")
    axis.grid(axis="y", alpha=0.3, zorder=0)
    axis.legend(
        handles=[
            Patch(facecolor=color, label=label)
            for color, label in zip(METRIC_COLORS, METRIC_LABELS)
        ]
        + [Patch(facecolor="white", edgecolor="black", label="Mean +/- 1 SD")],
        title="Metric / spread",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot trial error time series and scenario RMSE summaries."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder containing one subfolder per scenario (default: this script's folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure output folder (default: <data-dir>/plots).",
    )
    parser.add_argument(
        "--sample-trials",
        type=int,
        nargs="+",
        default=[0],
        metavar="TRIAL",
        help="Trial number(s) to plot for every scenario (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_dir / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = discover_scenarios(data_dir)

    for scenario, scenario_dir, _ in scenarios:
        for trial in args.sample_trials:
            plot_trial(scenario, trial, scenario_dir, output_dir)

    plot_rmse_summary(scenarios, output_dir / "rmse_summary.png")
    print(f"Saved plots for {len(scenarios)} scenarios to {output_dir}")


if __name__ == "__main__":
    main()
