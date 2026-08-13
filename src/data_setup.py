import random
import shutil
from pathlib import Path
from collections import defaultdict


class DataConfigutator:
    def __init__(
        self,
        images_path: str | Path,
        labels_path: str | Path,
        class_info_path: str | Path,
        n_folds: int = 5,
        random_seed: int = 55,
    ):

        self.images_path = Path(images_path)
        self.labels_path = Path(labels_path)
        self.class_info_path = Path(class_info_path)
        self.class_info = Path(class_info_path)
        self.n_folds = n_folds
        self.random_seed = random_seed

        self.files = self.processing_label_files(self.loading_label_files())
        self.folds = self.stratified_split(self.files)

    def write_config(
            self,
            target_path: str | Path,
            set_config: dict[str, dict],
            write_files: bool = True,
            overwrite: bool = False,
            ) -> None | dict[str, list[str]]:

        """
        Writes the dataset configuration to a YAML file with the matching .txt files.

        Args:
            target_path (str | Path): Path to save the configuration files.
            set_config (dict[str, dict]):
                Configurates the setup - provide a subset of ["train", "val", "test"] as keys.
                For each key, provide a dict with keys:
                    - folds (int | list[int]):
                        list of fold indices or a single fold index to include.
                    - noise_levels (int | list[int], optional):
                        Noise levels to include. Defaults to 20 (no noise).
                        set -1 for all noise levels.
            write_files (bool): If true, files will be written to disk, defaults to True.
            overwrite (bool): If true, existing files will be overwritten, defaults to False.

        Returns:
            None if write_files is True, else returns a dict with file contents.

        Writes:
            - A YAML configuration file at target_path/dataset.yaml
            - Corresponding .txt files for each set at target_path/{set_name}.txt

        Example:
            set_config = {
                "train": {
                    "folds": [0, 1, 2],
                    "noise_levels": -1
                    }
            }
        """

        set_config = self._validate_set_config(set_config)

        target_path = Path(target_path)
        files_to_write = defaultdict(list)
        files_to_write['data.yaml'] = []

        for set_name, config in set_config.items():
            line = f"{set_name}: {str(target_path / f'{set_name}.txt')}"
            files_to_write['data.yaml'].append(line)

            files_to_write[f'{set_name}.txt'] = self.get_file_list(
                folds=config["folds"],
                noise_levels=config["noise_levels"]
                )

        with open(self.class_info_path, 'r') as f:
            class_info = [line.rstrip() for line in f]  # preserve leading indentation
        files_to_write['data.yaml'].extend(class_info)

        if write_files:
            if target_path.exists():
                if overwrite:
                    print(f"Overwriting existing path: {target_path}")
                    shutil.rmtree(target_path)
                    target_path.mkdir(parents=True)
                else:
                    raise FileExistsError(f"Path already exists: {target_path}")
            else:
                target_path.mkdir(parents=True)

            for filename, lines in files_to_write.items():
                with open(target_path / filename, 'w') as f:
                    for line in lines:
                        f.write(line + '\n')

        else:
            return files_to_write

    def get_file_list(
            self,
            folds: list[int],
            noise_levels: list[int],
            suffix: str = ".png"
            ) -> list[str]:

        if suffix == ".png":
            base = self.images_path
        elif suffix == ".txt":
            base = self.labels_path
        else:
            raise ValueError(f"Invalid suffix: {suffix}. Must be '.png' or '.txt'.")

        file_list = []

        for fold_idx in folds:
            for sample_data in self.folds[fold_idx].values():
                for noise_level in noise_levels:
                    stem = sample_data[noise_level]
                    file_list.append(str(base / f"{stem}{suffix}"))

        return file_list

    def stratified_split(
            self,
            processed_files: dict[int, dict],
            ) -> dict[int, dict[int, dict]]:

        folds = {i: {} for i in range(self.n_folds)}

        class_to_samples = defaultdict(list)
        for sample_id, data in processed_files.items():
            class_to_samples[data["target"]].append(sample_id)

        rng = random.Random(self.random_seed)
        for samples in class_to_samples.values():
            rng.shuffle(samples)

        for samples in class_to_samples.values():
            for i, sample_id in enumerate(samples):
                fold_idx = i % self.n_folds
                folds[fold_idx][sample_id] = processed_files[sample_id]

        return folds

    def processing_label_files(
            self,
            raw_label_files: list[Path]
            ) -> dict[int, dict[int, str]]:

        """
        Processes the label files and groups them by sample ID and noise level.
        Also determines the target class for each sample based on the contents of the label files.
        (for no target: -1, single class: class ID, multiple classes: 99)
        """

        processed_files = defaultdict(dict)
        for file_path in raw_label_files:
            stem_parts = file_path.stem.split("_")
            sample_id = int(stem_parts[1][6:])
            noise_level = int(stem_parts[3][3:-2])
            processed_files[sample_id][noise_level] = file_path.stem

        for key, value in processed_files.items():
            processed_files[key] = dict(sorted(value.items()))
            with open(self.labels_path / f"{value[0]}.txt", 'r') as f:
                lines = f.readlines()
                if len(lines) == 0:
                    processed_files[key]["target"] = -1
                else:
                    ids_in_file = set()
                    for line in lines:
                        class_id = int(line.split()[0])
                        ids_in_file.add(class_id)
                    if len(ids_in_file) == 1:
                        processed_files[key]["target"] = ids_in_file.pop()
                    else:
                        processed_files[key]["target"] = 99

        return processed_files

    def loading_label_files(self) -> list[Path]:
        """
        Loads all label files and ensures that each label file has a corresponding image file.
        Returns a list of matched label file paths. Voids unmatched files with a warning.
        """
        image_stems = {f.stem for f in self.images_path.glob("*.png")}
        label_stems = {f.stem for f in self.labels_path.glob("*.txt")}

        matched_stems = image_stems & label_stems
        unmatched_stems = image_stems ^ label_stems

        if len(unmatched_stems) > 0:
            print(f"Warning: {len(unmatched_stems)} unmatched files found:")
            for stem in unmatched_stems:
                print(f" - {stem}")

        return [self.labels_path / f"{stem}.txt" for stem in sorted(matched_stems)]

    def _validate_set_config(
            self,
            set_config
            ):

        validated_set_config = defaultdict(dict)

        listed_folds = []
        for current_set, current_config in set_config.items():
            if current_set not in ["train", "val", "test"]:
                raise ValueError(f"Invalid set name: {current_set}. Must be 'train', 'val', or 'test'.")

            folds = current_config["folds"]
            if isinstance(folds, int):
                folds = [folds]
            # check if folds is subset of range(self.n_folds
            if not set(folds).issubset(set(range(self.n_folds))):
                raise ValueError(f"Invalid folds for set {current_set}: {folds}. Must be subset of [0, {self.n_folds - 1}].")
            listed_folds.extend(folds)
            validated_set_config[current_set]["folds"] = folds

            noise_levels = current_config.get("noise_levels", 20)
            if noise_levels == -1:
                noise_levels = list(range(-30, 20 + 1, 2))
            else:
                if isinstance(noise_levels, int):
                    noise_levels = [noise_levels]

                unique_noise_levels = set(noise_levels)

                if len(unique_noise_levels) != len(noise_levels):
                    raise ValueError(f"Duplicate noise levels Provided for set {current_set}: {noise_levels}.")

                if unique_noise_levels - set(range(-30, 20 + 1, 2)):
                    raise ValueError(f"Invalid noise levels for set {current_set}: {noise_levels}. Must be in range -30 to 20 dB with step 2.")

            validated_set_config[current_set]["noise_levels"] = noise_levels

        if len(listed_folds) != len(set(listed_folds)):
            raise ValueError("Folds overlap between different sets in set_config.")

        return validated_set_config
