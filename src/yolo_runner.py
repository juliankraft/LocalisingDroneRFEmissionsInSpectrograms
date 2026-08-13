import json
import shutil
from datetime import datetime
from pathlib import Path
from ultralytics.models.yolo import YOLO
from src.data_setup import DataConfigutator


TRAIN_KWARGS = {
    # Training settings
    "epochs": 100,              # Number of training epochs
    "imgsz": 640,               # Input image size
    "batch": -1,                # Batch size (-1 for auto)
    "patience": 20,             # Early stopping patience
    "save": True,               # Save checkpoints
    "device": 0,                # GPU device (use 'cpu' if no GPU)
    "workers": 8,               # Number of dataloader workers
    "exist_ok": True,           # Overwrite existing experiment
    "pretrained": True,         # Use pretrained weights
    "optimizer": "auto",        # Optimizer (auto, SGD, Adam, AdamW, etc.)
    "verbose": True,            # Verbose output
    "deterministic": True,      # Deterministic mode
    "single_cls": False,        # Train as single-class detector
    "rect": False,              # Rectangular training
    "cos_lr": False,            # Use cosine learning rate scheduler
    "resume": False,            # Resume training from last checkpoint
    "amp": True,                # Automatic Mixed Precision training
    # Learning rate settings
    "lr0": 0.01,                # Initial learning rate
    "lrf": 0.01,                # Final learning rate (lr0 * lrf)
    "momentum": 0.937,          # SGD momentum/Adam beta1
    "weight_decay": 0.0005,     # Optimizer weight decay
    "warmup_epochs": 3.0,       # Warmup epochs
    "warmup_momentum": 0.8,     # Warmup initial momentum
    "warmup_bias_lr": 0.1,      # Warmup initial bias lr
    # Data augmentation
    "hsv_h": 0.0,               # HSV-Hue augmentation
    "hsv_s": 0.0,               # HSV-Saturation augmentation
    "hsv_v": 0.0,               # HSV-Value augmentation
    "degrees": 0.0,             # Rotation augmentation
    "translate": 0.0,           # Translation augmentation
    "scale": 0.0,               # Scale augmentation
    "shear": 0.0,               # Shear augmentation
    "perspective": 0.0,         # Perspective augmentation
    "flipud": 0.5,              # Vertical flip probability
    "fliplr": 0.0,              # Horizontal flip probability
    "mosaic": 0.0,              # Mosaic augmentation probability
    "mixup": 0.0,               # Mixup augmentation probability
    "copy_paste": 0.0,          # Copy-paste augmentation probability
}


class YOLORunner:
    def __init__(
            self,
            image_path: str | Path,
            label_path: str | Path,
            class_info_path: str | Path,
            project_path: str | Path,
            model_name: str = "yolov8s.pt",
            n_folds: int = 5,
            crossval: bool = False,
            random_seed: int = 55,
            train_kwargs: dict | None = None,
            train_noise_levels: list[int] | int = 20,
            val_noise_levels: list[int] | int = 20,
            test_noise_levels: list[int] | int = 20,
            dev_mode: bool = False,
            save_testmetrics: bool = True,
            save_predictions: bool = False,
            append_mode: bool = False
    ):
        self.image_path = Path(image_path)
        self.label_path = Path(label_path)
        self.class_info_path = Path(class_info_path)
        self.project_path = Path(project_path)
        self.model_name = model_name
        self.n_folds = n_folds
        self.crossval = crossval
        self.random_seed = random_seed
        self.train_kwargs = {**TRAIN_KWARGS, **(train_kwargs or {})}
        self.dev_mode = dev_mode
        self.start_time = datetime.now()
        self.append_mode = append_mode

        if self.crossval and self.append_mode:
            raise ValueError("Append mode is not supported with cross-validation.")

        if dev_mode:
            self.train_kwargs["epochs"] = 1
            self.train_kwargs["patience"] = 1
            train_noise_levels = 20
            val_noise_levels = 20
            test_noise_levels = [18, 20]
            print("DEV MODE: epochs=1, patience=1, one noise levels (20dB)")
            print("for train, val, two noise levels (18dB, 20dB) for test")

        self.train_noise_levels = train_noise_levels
        self.val_noise_levels = val_noise_levels
        self.save_testmetrics = save_testmetrics
        self.save_predictions = save_predictions
        if test_noise_levels == -1:
            self.test_noise_levels = list(range(-30, 20 + 1, 2))
        elif isinstance(test_noise_levels, int):
            self.test_noise_levels = [test_noise_levels]
        else:
            self.test_noise_levels = test_noise_levels

        if self.append_mode and self.project_path.exists():
            for item in self.project_path.glob("test*"):
                shutil.rmtree(item) if item.is_dir() else item.unlink()
            train_dir = self.project_path / "train"
            if train_dir.exists():
                shutil.rmtree(train_dir)
        else:
            self.append_mode = False
            self.project_path.mkdir(parents=True, exist_ok=False)

        self.data = DataConfigutator(
            images_path=self.image_path,
            labels_path=self.label_path,
            class_info_path=self.class_info_path,
            n_folds=self.n_folds,
            random_seed=self.random_seed
            )

        self.fold_assignment = {
            "test": 0,
            "val": 1,
            "train": list(range(2, self.n_folds))
            }

        if self.append_mode:
            self.model_path = self.project_path / "model" / self.model_name
            self.best_model_path = self.project_path / "training" / "weights" / "best.pt"
        else:
            self.model_path = self._download_model()
            self.best_model_path = None

    def _download_model(self) -> Path:
        model_path = self.project_path / "model" / self.model_name
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model = YOLO(self.model_name)
        model.save(str(model_path))
        return model_path

    def train_fold(
            self,
            output_path: Path,
            ) -> None:
        """Train and evaluate a single fold."""

        set_config = {
            "train": {
                "folds": self.fold_assignment["train"],
                "noise_levels": self.train_noise_levels
                },
            "val": {
                "folds": self.fold_assignment["val"],
                "noise_levels": self.val_noise_levels
                }
            }

        # Write data config
        self.data.write_config(
            target_path=output_path / "train",
            set_config=set_config,
            )

        # Train
        model = YOLO(str(self.model_path))
        kwargs = self.train_kwargs.copy()
        kwargs["data"] = str(output_path / "train" / "data.yaml")
        kwargs["project"] = str(output_path)
        kwargs["name"] = "training"
        kwargs["seed"] = self.random_seed

        results = model.train(**kwargs)

        # Store best model path for testing
        if model.trainer and model.trainer.best:
            self.best_model_path = Path(model.trainer.best)
        else:
            raise RuntimeError("Training failed - no best model found.")

        return results

    def test_fold(
            self,
            output_path: Path,
            ) -> dict:
        """Test trained model on test fold with specified noise levels."""
        if self.best_model_path is None:
            raise RuntimeError("No trained model found. Run train_fold() first.")

        test_metrics = {}
        model = YOLO(str(self.best_model_path))

        for noise_level in self.test_noise_levels:
            test_path = output_path / f"test_{noise_level}db"

            # YOLO requires train/val keys even for testing, so we duplicate
            set_config = {
                "train": {
                    "folds": self.fold_assignment["train"],
                    "noise_levels": self.train_noise_levels
                    },
                "val": {
                    "folds": self.fold_assignment["val"],
                    "noise_levels": self.val_noise_levels
                    },
                "test": {
                    "folds": self.fold_assignment["test"],
                    "noise_levels": noise_level
                    }
                }

            # Write test data config
            self.data.write_config(
                target_path=test_path,
                set_config=set_config,
                )

            test_metrics[noise_level] = model.val(
                data=str(test_path / "data.yaml"),
                split="test",
                project=str(test_path),
                name="results",
                save_txt=self.save_predictions,
                save_conf=True,
                conf=0.25,
            )

        return test_metrics

    def run_only_one_fold(self) -> dict:
        """Run training and testing on a single fold without cross-validation."""
        if not self.append_mode:
            self.train_fold(output_path=self.project_path)
        test_metrics = self.test_fold(output_path=self.project_path)

        return self._process_test_metrics(test_metrics, crossval=False)

    def run_crossval(self) -> dict:
        """Run full cross-validation."""
        all_test_metrics = {}

        for fold_idx in range(self.n_folds):
            self.train_fold(output_path=self.project_path / f"fold_{fold_idx}")
            all_test_metrics[fold_idx] = self.test_fold(output_path=self.project_path / f"fold_{fold_idx}")
            self._rotate_folds()

        return self._process_test_metrics(all_test_metrics, crossval=True)

    def run(self) -> dict:
        """Run training based on crossval setting."""
        if self.crossval:
            return self.run_crossval()
        else:
            return self.run_only_one_fold()

    def _rotate_folds(self) -> None:
        """Rotate folds for different train/val/test splits."""
        self.fold_assignment = {
            "test": (self.fold_assignment["test"] + 1) % self.n_folds,
            "val": (self.fold_assignment["val"] + 1) % self.n_folds,
            "train": [
                (fold + 1) % self.n_folds for fold in self.fold_assignment["train"]
            ]
            }

    def _process_test_metrics(
            self,
            test_metrics: dict,
            crossval: bool = False
            ) -> dict:
        """Save test metrics to JSON file.

        Args:
            test_metrics: For single fold: {noise_level: metrics}
                          For crossval: {fold_idx: {noise_level: metrics}}
            crossval: Whether this is crossval (affects output structure)
        """

        elapsed_time = datetime.now() - self.start_time

        def extract_metrics(metrics):
            return {
                'mAP50': metrics.box.map50,
                'mAP50-95': metrics.box.map
            }

        if crossval:
            output = {
                'seed': self.random_seed,
                'n_folds': self.n_folds,
                'folds': {
                    str(fold_idx): {
                        str(noise_level): extract_metrics(metrics)
                        for noise_level, metrics in fold_metrics.items()
                    }
                    for fold_idx, fold_metrics in test_metrics.items()
                }
            }
        else:
            output = {
                'seed': self.random_seed,
                'test_metrics': {
                    str(noise_level): extract_metrics(metrics)
                    for noise_level, metrics in test_metrics.items()
                }
            }

        output['elapsed_time_seconds'] = elapsed_time.total_seconds()
        output['settings'] = {
            'model_name': self.model_name,
            'n_folds': self.n_folds,
            'crossval': self.crossval,
            'dev_mode': self.dev_mode,
            'train_noise_levels': self.train_noise_levels,
            'val_noise_levels': self.val_noise_levels,
            'test_noise_levels': self.test_noise_levels,
            'image_path': str(self.image_path),
            'label_path': str(self.label_path),
            'class_info_path': str(self.class_info_path),
            'project_path': str(self.project_path),
            'yolo_kwargs': self.train_kwargs,
        }

        if self.save_testmetrics:
            metrics_file = self.project_path / 'test_metrics.json'

            with open(metrics_file, 'w') as f:
                json.dump(output, f, indent=2)

            print(f"Test metrics saved to: {metrics_file}")

        return output


def prediction_writer(
        images_txt: Path | str,
        model_path: Path | str,
        output_path: Path | str | None = None
        ) -> list:
    """Run YOLO inference on a list of images.

    Args:
        images_txt: Path to a text file with one image path per line.
        model_path: Path to the YOLO model weights (.pt file).
        output_path: Directory to save prediction labels (txt format).
                     If None, results are returned but not saved to disk.

    Returns:
        List of YOLO Results objects, one per image.
    """
    model_path = Path(model_path)

    if output_path is not None:
        save_txt = True
        output_path = Path(output_path)
        project = output_path.parent
        name = output_path.name
    else:
        save_txt = False
        project = None
        name = None

    model = YOLO(str(model_path))

    with open(images_txt, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]

    chunk_size = 256
    results = []
    for i in range(0, len(image_paths), chunk_size):
        chunk = image_paths[i:i + chunk_size]
        results.extend(model.predict(
            source=chunk,
            project=project,
            name=name,
            save_txt=save_txt,
            save_conf=True,
            exist_ok=True,
            stream=True,
        ))

    return results
