import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

N_SAMPLES_VALUES = [0, 2, 4, 8, 16]
BACKBONES = ["GCN", "GAT", "GIN", "GraphGPS", "GrokFormer", "DualFormer"]
BACKBONE_COLORS = {
    "GCN":        "#59A14F",  # green
    "GAT":        "#6FA8DC",  # blue
    "GIN":        "#76B7B2",  # teal
    "GraphGPS":   "#E6807F",  # coral
    "GrokFormer": "#F1A340",  # orange
    "DualFormer": "#B07AA1",  # purple
}
BACKBONE_MARKERS = {
    "GCN": "o", "GAT": "s", "GIN": "^",
    "GraphGPS": "D", "GrokFormer": "v", "DualFormer": "P",
}
BAND_ALPHA = 0.12  
BACKBONE_FAMILIES = {
    "mpnn": ["GCN", "GAT", "GIN"],
    "gt":   ["GraphGPS", "GrokFormer", "DualFormer"],
}
DATASETS = ["Graph_SST2", "SPMotif_b_05", "SPMotif_b_07", "SPMotif_b_09"]
DATASET_LABELS = {
    "Graph_SST2":   "G-SST2",
    "SPMotif_b_05": "SpM-0.5",
    "SPMotif_b_07": "SpM-0.7",
    "SPMotif_b_09": "SpM-0.9",
}
FIGURES = [
    ("test_auprc", "AUPRC", "auprc", (0.0, 1.0)),
]
SUMMARY_COLS = [
    "av_test_auroc", "stdev_test_auroc",
    "av_test_auprc", "stdev_test_auprc",
    "av_test_accuracy", "stdev_test_accuracy",
    "av_train_time_sec", "stdev_train_time_sec",
    "av_train_time_per_epoch_sec", "av_epochs_trained",
]
plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 1.0,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})


def csv_path(results_dir, dataset, backbone, n):
    d = os.path.join(results_dir, dataset, "integrated")
    if n == 0:
        return os.path.join(d, f"CIF_J_NoMC_{backbone}_encoder_integrated.csv")
    if n == 16:
        swept = os.path.join(d, f"CIF_{backbone}_encoder_nsamples16_integrated.csv")
        return swept if os.path.exists(swept) else os.path.join(
            d, f"CIF_{backbone}_encoder_integrated.csv")
    return os.path.join(d, f"CIF_{backbone}_encoder_nsamples{n}_integrated.csv")


def read_row(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def get_val(row, col):
    if row is None or col not in row:
        return None
    v = row[col]
    if v in ("", "nan", "NaN", None):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_series(results_dir, dataset, backbone, metric):
    xs, means, stds = [], [], []
    for i, n in enumerate(N_SAMPLES_VALUES):
        row = read_row(csv_path(results_dir, dataset, backbone, n))
        mean = get_val(row, f"av_{metric}")
        std = get_val(row, f"stdev_{metric}")
        if mean is None:
            continue
        xs.append(i)
        means.append(mean)
        stds.append(std if std is not None else 0.0)
    return np.array(xs), np.array(means), np.array(stds)


def plot_faceted(results_dir, metric, ylabel, tag, output_dir, ylim=None, backbones=None):
    if backbones is None:
        backbones = BACKBONES
    x = np.arange(len(N_SAMPLES_VALUES))
    fig, axes = plt.subplots(
        1, len(DATASETS), figsize=(4.1 * len(DATASETS), 4.3), squeeze=False
    )
    axes = axes[0]
    any_data = False

    for ax, dataset in zip(axes, DATASETS):
        panel_has_data = False
        for backbone in backbones:
            xs, means, stds = load_series(results_dir, dataset, backbone, metric)
            if xs.size == 0:
                continue
            panel_has_data = any_data = True
            color = BACKBONE_COLORS[backbone]
            ax.fill_between(xs, means - stds, means + stds,
                            color=color, alpha=BAND_ALPHA, linewidth=0, zorder=1)
            ax.errorbar(xs, means, yerr=stds, color=color,
                        marker=BACKBONE_MARKERS[backbone], markersize=8,
                        markeredgecolor="white", markeredgewidth=0.9,
                        linewidth=2.2, capsize=3, elinewidth=1.2, capthick=1.2,
                        zorder=3, label=backbone)

        ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in N_SAMPLES_VALUES], fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.margins(x=0.08)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if not panel_has_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=12)

    if not any_data:
        print(f"  [skip] no data for '{metric}' figure")
        plt.close(fig)
        return

    handles = [
        Line2D([0], [0], color=BACKBONE_COLORS[b], marker=BACKBONE_MARKERS[b],
               markeredgecolor="white", markeredgewidth=0.9, markersize=8,
               linewidth=2.2, label=b)
        for b in backbones
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(backbones),
               fontsize=12, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.supxlabel("N_SAMPLES", fontsize=14)
    fig.supylabel(ylabel, fontsize=14)
    fig.tight_layout(rect=[0.02, 0.03, 1, 0.93])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"sensitivity_{tag}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def write_summary(results_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "sensitivity_summary.csv")
    header = ["backbone", "dataset", "n_samples"] + SUMMARY_COLS
    n_rows = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for backbone in BACKBONES:
            for dataset in DATASETS:
                for n in N_SAMPLES_VALUES:
                    row = read_row(csv_path(results_dir, dataset, backbone, n))
                    if row is None:
                        continue
                    writer.writerow(
                        [backbone, dataset, n] + [row.get(c, "") for c in SUMMARY_COLS]
                    )
                    n_rows += 1
    print(f"Saved: {out_path}  ({n_rows} configurations)")


def main():
    parser = argparse.ArgumentParser(description="Plot the CIF N_SAMPLES sensitivity analysis")
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--output-dir", default="../results/sensitivity_plots")
    parser.add_argument("--split-families", action="store_true",
                        help="one figure per backbone family (MPNNs, GTs) with 3 lines each, "
                             "instead of a single 6-line figure")
    args = parser.parse_args()

    for metric, ylabel, tag, ylim in FIGURES:
        if args.split_families:
            for family, fam_backbones in BACKBONE_FAMILIES.items():
                plot_faceted(args.results_dir, metric, ylabel, f"{tag}_{family}", args.output_dir,
                             ylim=ylim, backbones=fam_backbones)
        else:
            plot_faceted(args.results_dir, metric, ylabel, tag, args.output_dir,
                         ylim=ylim, backbones=BACKBONES)

    write_summary(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
