import argparse
import json
from pathlib import Path
from src.yolo_runner import YOLORunner

REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO while gradually adding noise levels to the training set.")
    parser.add_argument(
        "--dataset-path", type=Path, required=True,
        help="Path to the yolo-ready dataset (containing images/, labels/, data.txt).",
        )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="CUDA device to train on, e.g. '0' or 'cpu' (default: 'cpu').",
        )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset_path = args.dataset_path
    project_path = REPO_ROOT / "output" / "train_yolo_adding_noise"

    if project_path.exists():
        raise FileExistsError(
            f"Project path already exists: {project_path}\n"
            "Refusing to overwrite a possibly completed training run. Remove it manually if you want to retrain."
        )

    metrics = {}
    train_noise = []

    for i, x in enumerate(range(20, -30 - 1, -2)):
        train_noise.insert(0, x)

        target_path = project_path / f"train_on_{i + 1}_noise_levels"

        runner = YOLORunner(
            image_path=dataset_path / "images",
            label_path=dataset_path / "labels",
            class_info_path=dataset_path / "data.txt",
            project_path=target_path,
            model_name="yolov8s.pt",
            n_folds=5,
            crossval=False,
            random_seed=55,
            train_noise_levels=train_noise,
            val_noise_levels=train_noise,
            test_noise_levels=-1,  # test on all noise levels
            dev_mode=False,
            save_testmetrics=True,
            save_predictions=True,
            append_mode=False,
            train_kwargs={"device": args.device},
        )

        test_metrics = runner.run()
        metrics[f"train_on_{i + 1}_noise_levels"] = test_metrics

    with open(project_path / "collected_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
