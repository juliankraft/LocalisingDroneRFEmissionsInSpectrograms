# %%
import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.utils import eval_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visualizations for the train_yolo and train_yolo_adding_noise experiments.")
    parser.add_argument(
        "--dataset-path", type=Path, required=True,
        help="Path to the yolo-ready dataset (containing images/, labels/, data.txt) — same as passed to the train_*.py scripts.",
    )
    return parser.parse_args()


args = parse_args()
dataset_path = args.dataset_path

REPO_ROOT = Path(__file__).resolve().parent
train_yolo_experiment = REPO_ROOT / "output" / "train_yolo"
train_yolo_metrics = train_yolo_experiment / "test_metrics.json"
train_yolo_adding_noise_experiment = REPO_ROOT / "output" / "train_yolo_adding_noise"
train_yolo_adding_noise_metrics = train_yolo_adding_noise_experiment / "collected_metrics.json"
output_path = REPO_ROOT / "output" / "visualizations"
output_path.mkdir(parents=True, exist_ok=True)

snr_levels = list(range(-30, 20 + 1, 2))  # X axis: -30 to 20


###############################################################################
# plotting mAP vs SNR across folds
# %% ##########################################################################
try:
    metrics = json.loads(train_yolo_metrics.read_text())
    fig, ax = plt.subplots(figsize=(10, 6))

    for fold_idx in range(5):
        fold_data = metrics["folds"][str(fold_idx)]

        map50 = [fold_data[str(snr)]["mAP50"] for snr in snr_levels]
        map50_95 = [fold_data[str(snr)]["mAP50-95"] for snr in snr_levels]

        label_50 = "mAP50" if fold_idx == 0 else None
        label_95 = "mAP50-95" if fold_idx == 0 else None

        ax.plot(snr_levels, map50, color="#5a8fc7", label=label_50)
        ax.plot(snr_levels, map50_95, color="#c75a5a", linestyle="--", label=label_95)

    ax.invert_xaxis()
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("mAP")
    # ax.set_title("mAP vs SNR across folds")
    ax.legend()

    for spine in ax.spines.values():
        spine.set_color("gray")

    plt.tight_layout()
    plt.savefig(output_path / "map50_vs_snr_across_folds.png", dpi=300)
    plt.show()
except FileNotFoundError as e:
    print(f"Skipping mAP vs SNR across folds (train_yolo experiment not found): {e}")

###############################################################################
#  3D plot of mAP50 vs SNR and number of noise levels in training set
# %% ##########################################################################
try:
    metrics = json.loads(train_yolo_adding_noise_metrics.read_text())

    # Extract data for 3D plot
    train_levels = sorted([int(k.split('_')[2]) for k in metrics.keys()])  # Y axis: n from train_on_n_noise_levels

    # Create meshgrid
    X, Y = np.meshgrid(snr_levels, train_levels)
    Z = np.zeros_like(X, dtype=float)

    # Fill Z values (mAP50)
    for i, n in enumerate(train_levels):
        key = f"train_on_{n}_noise_levels"
        test_metrics = metrics[key]["test_metrics"]
        for j, snr in enumerate(snr_levels):
            Z[i, j] = test_metrics[str(snr)]["mAP50"]

    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

    ax.set_xlabel('SNR (dB) in Test Set')
    ax.set_ylabel('n Train Noise Levels in the Training Set')
    ax.set_yticks(range(1, len(train_levels) + 1, 2))
    ax.set_zlabel('mAP50')
    # ax.set_title('mAP50 vs SNR and Training Noise Diversity')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='mAP50')

    plt.tight_layout()
    plt.savefig(output_path / "3d_map50_vs_snr_and_train_noise_levels.png", dpi=300)
    plt.show()
except FileNotFoundError as e:
    print(f"Skipping 3D mAP50 plot (train_yolo_adding_noise experiment not found): {e}")

###############################################################################
# map50 and map50-95 vs SNR for different training noise diversities
# %% ##########################################################################
try:
    metrics = json.loads(train_yolo_adding_noise_metrics.read_text())
    noise_levels = [1, 26]
    designations = ["trained SNR = 20dB", "trained SNR = -30dB to 20dB"]

    colors = matplotlib.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, n in enumerate(noise_levels):
        color = colors[i % len(colors)]
        test_metrics_n = metrics[f"train_on_{n}_noise_levels"]["test_metrics"]

        map50 = [test_metrics_n[str(snr)]["mAP50"] for snr in snr_levels]
        map50_95 = [test_metrics_n[str(snr)]["mAP50-95"] for snr in snr_levels]

        ax.plot(snr_levels, map50, color=color, linestyle="--")
        ax.plot(snr_levels, map50_95, color=color, linestyle=":")

    ax.invert_xaxis()
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("mAP")
    ax.grid(True, linestyle='--', alpha=0.5)
    # ax.set_title("mAP vs SNR by training noise diversity")

    color_handles = [Patch(color=colors[i % len(colors)], label=f"{designations[i]}") for i, n in enumerate(noise_levels)]
    style_handles = [
        Line2D([0], [0], color="gray", linestyle="--", label="mAP50"),
        Line2D([0], [0], color="gray", linestyle=":", label="mAP50-95"),
    ]
    ax.legend(handles=color_handles + [Line2D([], [], color="none")] + style_handles, loc="lower left")

    for spine in ax.spines.values():
        spine.set_color("gray")

    plt.tight_layout()
    plt.savefig(output_path / "map_vs_snr_by_noise_levels.png", dpi=600)
    plt.show()
except FileNotFoundError as e:
    print(f"Skipping mAP vs SNR by noise diversity (train_yolo_adding_noise experiment not found): {e}")


###############################################################################
# confusion matrix and multi-panel plot
# %% ##########################################################################
def build_confusion_matrix(eval_h):
    df = eval_h.results_df
    n = len(eval_h.names)
    bg = n
    labels = list(eval_h.names.values()) + ["background"]

    cm = np.zeros((n + 1, n + 1), dtype=float)

    matched = df[df["result"].isin(["TP", "WC"])]
    np.add.at(cm,  # type: ignore[call-overload]
              (matched["pred_class_id"].astype(int).to_numpy(),
               matched["gt_class_id"].astype(int).to_numpy()), 1)

    fn_rows = df[df["result"] == "FN"]
    np.add.at(cm,  # type: ignore[call-overload]
              (np.full(len(fn_rows), bg, dtype=int),
               fn_rows["gt_class_id"].astype(int).to_numpy()), 1)

    fp_rows = df[df["result"] == "FP"]
    np.add.at(cm,  # type: ignore[call-overload]
              (fp_rows["pred_class_id"].astype(int).to_numpy(),
               np.full(len(fp_rows), bg, dtype=int)), 1)

    return cm, labels


try:
    cm_configs = [
        dict(experiment_path=train_yolo_experiment / "fold_0", test_noise_snr=20,  title="SNR 20 dB"),
        dict(experiment_path=train_yolo_experiment / "fold_0", test_noise_snr=-30, title="SNR -30 dB"),
    ]

    cms, labels_list, titles = [], [], []
    for cfg in cm_configs:
        ev = eval_helper(
            experiment_path=cfg["experiment_path"],
            data_path=dataset_path,
            test_noise_snr=cfg["test_noise_snr"],
            iou_threshold=0.50,
            conf_threshold=0.25,
        )
        cm, labels = build_confusion_matrix(ev)
        cms.append(cm)
        labels_list.append(labels)
        titles.append(cfg["title"])

    vmax = max(cm.max() for cm in cms)

    fig, axes = plt.subplots(1, len(cms), figsize=(7 * len(cms), 6), constrained_layout=True)
    for ax, cm, labels, title in zip(axes, cms, labels_list, titles):
        n = len(labels) - 1
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(n + 1):
            for j in range(n + 1):
                val = cm[i, j]
                if val > 0:
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            color="white" if val > vmax * 0.5 else "black", fontsize=9)
        ax.set_xticks(range(n + 1))
        ax.set_yticks(range(n + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title)

    fig.colorbar(im, ax=axes.tolist(), shrink=0.8)

    plt.savefig(output_path / "confusion_matrices.png", dpi=300)
except FileNotFoundError as e:
    print(f"Skipping confusion matrices (train_yolo experiment not found): {e}")

###############################################################################
# table of metrics for crossval experiment at SNR 20
# %% ##########################################################################

try:
    testSNR = 20
    cols = ["precision", "recall", "mAP50", "mAP50-95"]
    metrics = pd.DataFrame()
    yolo_metrics = json.loads(train_yolo_metrics.read_text())
    for fold_idx in range(5):
        fold = eval_helper(
            experiment_path=train_yolo_experiment / f"fold_{fold_idx}",
            data_path=dataset_path,
            test_noise_snr=testSNR,
            iou_threshold=0.50,
            conf_threshold=0.25,
            )

        fold_metrics = fold.compute_metrics(per_class=False)
        for key in ["mAP50", "mAP50-95"]:
            fold_metrics[key] = yolo_metrics["folds"][str(fold_idx)][str(testSNR)][key]
        metrics = pd.concat([metrics, pd.DataFrame([{c: fold_metrics[c] for c in cols}])], ignore_index=True)

    metrics.index = pd.RangeIndex(1, len(metrics) + 1, name="fold")
    metrics.loc["mean"] = metrics[cols].mean()
    metrics.loc["± std"] = metrics[cols].std()

    print(metrics.round(4))
    metrics.round(4).to_csv(output_path / "train_yolo_metrics_table.csv")
except FileNotFoundError as e:
    print(f"Skipping metrics table (train_yolo experiment not found): {e}")

