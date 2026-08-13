# Drone RF Object Detection

Training code for a YOLOv8-based detector that localises drone remote-control RF emissions in spectrograms. This repository accompanies the paper:

> Localising Drone RF Emissions in Spectrograms: Automatic Annotation and Noise-Robust Object Detection
> Stefan Glüge, Julian Kraft, Christof Schüpbach, Matthias Nyfeler
> NCTA 2026

- **Dataset:** https://doi.org/10.5281/zenodo.21428081

## What this is

Two training scripts built on top of `src/`, a small package wrapping [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) with the noise-level/fold logic used in the paper:

- `train_yolo.py` — trains a YOLOv8s detector with 5-fold cross-validation across all noise levels.
- `train_yolo_adding_noise.py` — trains repeatedly while gradually widening the training noise-level range, to study noise-robustness.

## Usage

Run from the repository root:

```bash
conda env create -f drone_env.yml
conda activate drone_rf

python train_yolo.py --dataset-path <path/to/yolo_ready_dataset>
python train_yolo_adding_noise.py --dataset-path <path/to/yolo_ready_dataset>
```

Unpack the [dataset](https://doi.org/10.5281/zenodo.21428081) and pass its
location as `--dataset-path`. It must be a directory laid out with `images/`,
`labels/`, and `data.txt`.

Both scripts train on CPU by default. To use a GPU, pass its `nvidia-smi`
index with `--device`, e.g. `--device 0`.

Each script writes its results under `output/` in the repository root
(`output/train_yolo` and `output/train_yolo_adding_noise` respectively).

Once training has finished, generate plots and tables for whichever
experiment(s) are present under `output/`:

```bash
python evaluate.py --dataset-path <path/to/yolo_ready_dataset>
```

Results are written to `output/visualizations/`.

## Citation

If you use this code in your research, please cite the accompanying paper:

```bibtex
@article{gluge2026localising,
  title={Localising Drone RF Emissions in Spectrograms: Automatic Annotation and Noise-Robust Object Detection},
  author={Glüge, S. and Kraft, J. and Nyfeler, M. and Schüpbach, C.},
  year={2026}
}
```

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
You are free to share and adapt the material for any purpose, even commercially, as long as you provide appropriate credit, indicate if changes were made, and do not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
