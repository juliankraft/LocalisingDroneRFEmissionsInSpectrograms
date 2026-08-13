import json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib.figure import Figure


def render_detections(
    img: Image.Image,
    gt_bboxes: list[tuple],
    pred_bboxes: list[tuple],
    pred_results: list[str] | None = None,
    target_h_px: int = 600,
    dpi: int = 100,
    ) -> Figure:

    w, h = img.size
    fig_h = target_h_px / dpi
    fig = Figure(figsize=(fig_h * w / h, fig_h))
    ax = fig.add_subplot(111)
    ax.imshow(np.array(img.convert("L")), cmap="gray", aspect="auto", origin="upper")

    for i, (x1, y1, x2, y2) in enumerate(pred_bboxes):
        result = pred_results[i] if pred_results else None
        color = "orange" if result == "WC" else "red"
        ax.add_patch(mpatches.Rectangle(
            (x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
            linewidth=1, edgecolor=color, facecolor="none",
        ))

    for x1, y1, x2, y2 in gt_bboxes:
        ax.add_patch(mpatches.Rectangle(
            (x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
            linewidth=1, edgecolor="green", facecolor="none", linestyle="--",
        ))

    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


class eval_helper:
    def __init__(
            self,
            experiment_path: Path | str,
            data_path: Path | str,
            experiment_metrics: str | Path | None = None,
            test_noise_snr: int = 20,
            iou_threshold: float = 0.5,
            conf_threshold: float = 0.25,
            verbose: bool = False,
            ):

        self.test_noise_snr = test_noise_snr
        self.iou_threshold = iou_threshold
        if conf_threshold < 0.25:
            raise ValueError(
                f"conf_threshold {conf_threshold} is below 0.25;"
                "predictions are filtered at 0.25 during inference so lower values have no effect."
                )
        self.conf_threshold = conf_threshold

        if experiment_metrics is not None:
            self.metrics = json.loads(Path(experiment_metrics).read_text())
        else:
            self.metrics = {}

        self.data_path = Path(data_path)
        self.names = yaml.safe_load((self.data_path / "data.txt").read_text())["names"] 
        self.experiment_path = Path(experiment_path)
        self.model_path = self.experiment_path / "training" / "weights" / "best.pt"

        self.test_output_path = self.experiment_path / f"test_{self.test_noise_snr}db"

        self.test_samples = self.load_test_samples(self.test_output_path / "test.txt")

        self.prediction_path = self.test_output_path / "results" / "labels"
        if not self.prediction_path.exists():
            raise FileNotFoundError(
                f"Prediction path not found: {self.prediction_path}\n"
                "Re-run YOLORunner with save_predictions=True, or set use_test_predictions=False."
                )

        self.results_df = pd.concat([self.process_sample(idx) for idx in range(len(self))], ignore_index=True)

        counts = (
            self.results_df.groupby("sample")["result"]
            .value_counts()
            .unstack(fill_value=0)
            .reindex(columns=["TP", "FP", "FN", "WC"], fill_value=0)
            .rename(columns={"TP": "nTP", "FP": "nFP", "FN": "nFN", "WC": "nWC"})
            .reset_index()
            )
        counts["nER"] = counts["nFP"] + counts["nFN"] + counts["nWC"]

        gt_class = (
            self.results_df
            .dropna(subset=["gt_class_id"])
            .groupby("sample")["gt_class_id"]
            .first()
            .astype("Int64")
            .reset_index()
        )
        counts = counts.merge(gt_class, on="sample", how="left")
        counts["gt_class_id"] = counts["gt_class_id"].fillna(-1).astype(int)

        self.samples_overview = (
            counts
            .sort_values("nER", ascending=False)
            .reset_index(drop=True)
            )

        if verbose:
            e = self.results_df.groupby("result").size()
            n = {k: int(e.get(k, 0)) for k in ("TP", "FP", "FN", "WC")}
            n_err = int((self.samples_overview["nER"] > 0).sum())
            n_total = len(self.samples_overview)
            print(f"Experiment : {self.experiment_path.name}")
            print(f"SNR        : {self.test_noise_snr} dB  |  conf ≥ {self.conf_threshold}  |  IoU ≥ {self.iou_threshold}")
            print(f"Samples    : {n_total} total  |  {n_err} with errors  |  {n_total - n_err} clean")
            print(f"Detections : {sum(n.values())} total  |  TP {n['TP']}  FP {n['FP']}  FN {n['FN']}  WC {n['WC']}")

    def __len__(self):
        return len(self.test_samples)

    def __iter__(self):
        for sample in self.test_samples:
            yield sample

    def __getitem__(self, idx):
        sample = self.test_samples[idx]
        predictions, labels = self.load_sample(sample)
        return {
            "sample": sample,
            "predictions": predictions,
            "labels": labels,
            "results": self.results_df[self.results_df["sample"] == sample],
            "image": Image.open(self.data_path / "images" / f"{sample}.png"),
            }

    def load_test_samples(
            self,
            test_config_path: Path
            ) -> list[str]:
        test_samples = []
        with open(test_config_path, 'r') as f:
            for line in f:
                sample = Path(line.strip())
                test_samples.append(sample.stem)
        return sorted(test_samples)

    def load_file(
            self,
            path: Path
            ) -> list[dict]:
        if not path.exists():
            return []
        lines = path.read_text().splitlines()

        result = []
        for line in lines:
            split = line.split()
            entry = {}

            entry["class"] = int(split[0])

            cx, cy, w, h = float(split[1]), float(split[2]), float(split[3]), float(split[4])
            entry["bbox"] = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

            if len(split) > 5:
                entry["conf"] = float(split[5])
                if entry["conf"] < self.conf_threshold:
                    continue
            result.append(entry)

        return result

    def image_path(self, sample: str) -> Path:
        return self.data_path / "images" / f"{sample}.png"

    def load_sample(
            self,
            sample: str
            ) -> tuple[list[dict], list[dict]]:
        prediction_file = self.prediction_path / f"{sample}.txt"
        label_file = self.data_path / "labels" / f"{sample}.txt"

        return self.load_file(prediction_file), self.load_file(label_file)

    def _iou(self, b1: tuple, b2: tuple) -> float:
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return inter / union if union > 0 else 0.0

    def plot_sample(self, sample: str):
        rows = self.results_df[self.results_df["sample"] == sample]
        img = Image.open(self.image_path(sample))
        pred_rows = rows.dropna(subset=["pred_bbox"])
        return render_detections(
            img,
            gt_bboxes=rows["gt_bbox"].dropna().tolist(),
            pred_bboxes=pred_rows["pred_bbox"].tolist(),
            pred_results=pred_rows["result"].tolist(),
        )

    def process_sample(self, idx: int) -> pd.DataFrame:
        sample = self.test_samples[idx]
        pred, gt = self.load_sample(sample)

        matched_pred: set[int] = set()
        matched_gt: set[int] = set()
        matches: list[tuple[int, int, float]] = []

        if pred and gt:
            # IoU matrix: rows = gt, cols = pred (class-agnostic, matching YOLO confusion matrix)
            iou_matrix = np.zeros((len(gt), len(pred)))
            for j, g in enumerate(gt):
                for i, p in enumerate(pred):
                    iou_matrix[j, i] = self._iou(p["bbox"], g["bbox"])

            gt_idx, pred_idx = np.where(iou_matrix >= self.iou_threshold)
            if len(gt_idx):
                iou_vals = iou_matrix[gt_idx, pred_idx]
                order = iou_vals.argsort()[::-1]
                gt_idx, pred_idx, iou_vals = gt_idx[order], pred_idx[order], iou_vals[order]

                # Keep best-IoU match per prediction, then per GT
                _, keep = np.unique(pred_idx, return_index=True)
                gt_idx, pred_idx, iou_vals = gt_idx[keep], pred_idx[keep], iou_vals[keep]

                order = iou_vals.argsort()[::-1]
                gt_idx, pred_idx, iou_vals = gt_idx[order], pred_idx[order], iou_vals[order]
                _, keep = np.unique(gt_idx, return_index=True)
                gt_idx, pred_idx, iou_vals = gt_idx[keep], pred_idx[keep], iou_vals[keep]

                for gi, pi, iou in zip(gt_idx.tolist(), pred_idx.tolist(), iou_vals.tolist()):
                    matches.append((pi, gi, iou))
                    matched_pred.add(pi)
                    matched_gt.add(gi)

        rows = []

        for pi, gi, iou in matches:
            p, g = pred[pi], gt[gi]
            result = "TP" if p["class"] == g["class"] else "WC"
            rows.append(dict(
                sample=sample,
                result=result,
                pred_class_id=p["class"],
                gt_class_id=g["class"],
                iou=iou,
                conf=p.get("conf", np.nan),
                pred_bbox=p["bbox"],
                gt_bbox=g["bbox"],
                ))

        for i, p in enumerate(pred):
            if i in matched_pred:
                continue
            rows.append(dict(
                sample=sample,
                result="FP",
                pred_class_id=p["class"],
                gt_class_id=np.nan,
                iou=np.nan,
                conf=p.get("conf", np.nan),
                pred_bbox=p["bbox"],
                gt_bbox=None,
                ))

        for j, g in enumerate(gt):
            if j in matched_gt:
                continue
            rows.append(dict(
                sample=sample,
                result="FN",
                pred_class_id=np.nan,
                gt_class_id=g["class"],
                iou=np.nan,
                conf=np.nan,
                pred_bbox=None,
                gt_bbox=g["bbox"],
                ))

        cols = ["sample", "result", "pred_class_id", "gt_class_id", "iou", "conf", "pred_bbox", "gt_bbox"]
        df = pd.DataFrame(rows, columns=cols)
        df["iou"] = df["iou"].astype(float)
        df["conf"] = df["conf"].astype(float)
        return df

    def compute_metrics(self, per_class: bool = False) -> dict:
        df = self.results_df

        def _div(a, b):
            return float(a / b) if b > 0 else float("nan")

        def _metrics(tp, fp, fn_pure, wc_as_fp, wc_as_fn) -> dict:
            fn = fn_pure + wc_as_fn
            return {
                "precision":                _div(tp, tp + fp),
                "recall":                   _div(tp, tp + fn),
                "detection_recall":         _div(tp + wc_as_fn, tp + wc_as_fn + fn_pure),
                "classification_precision": _div(tp, tp + wc_as_fp),
            }

        is_tp = df["result"] == "TP"
        is_fp = df["result"].isin(["FP", "WC"])
        is_fn_pure = df["result"] == "FN"
        is_wc = df["result"] == "WC"

        out = _metrics(
            tp=is_tp.sum(),
            fp=is_fp.sum(),
            fn_pure=is_fn_pure.sum(),
            wc_as_fp=is_wc.sum(),
            wc_as_fn=is_wc.sum(),
        )

        if per_class:
            out["per_class"] = {}
            for class_id in self.names:
                pred_is_c = df["pred_class_id"] == class_id
                gt_is_c = df["gt_class_id"] == class_id
                out["per_class"][self.names[class_id]] = _metrics(
                    tp=(is_tp & pred_is_c).sum(),
                    fp=(is_fp & pred_is_c).sum(),
                    fn_pure=(is_fn_pure & gt_is_c).sum(),
                    wc_as_fp=(is_wc & pred_is_c).sum(),
                    wc_as_fn=(is_wc & gt_is_c).sum(),
                )

        return out

