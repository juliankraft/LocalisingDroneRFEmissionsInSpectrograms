import argparse
from pathlib import Path
from src.yolo_runner import YOLORunner

REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO with cross-validation over all noise levels.")
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
    project_path = REPO_ROOT / "output" / "train_yolo"

    if project_path.exists():
        raise FileExistsError(
            f"Project path already exists: {project_path}\n"
            "Refusing to overwrite a possibly completed training run. Remove it manually if you want to retrain."
        )

    runner = YOLORunner(
        image_path=dataset_path / "images",
        label_path=dataset_path / "labels",
        class_info_path=dataset_path / "data.txt",
        project_path=project_path,
        model_name="yolov8s.pt",
        n_folds=5,
        crossval=True,
        random_seed=55,
        train_noise_levels=list(range(-30, 20 + 1, 2)),
        val_noise_levels=list(range(-30, 20 + 1, 2)),
        test_noise_levels=-1,
        dev_mode=False,
        save_testmetrics=True,
        save_predictions=True,
        train_kwargs={"device": args.device},
    )

    runner.run()
