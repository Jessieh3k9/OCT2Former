import os

import numpy as np
from torch.utils.data import Dataset

from .transform import convert_to_tensor, fetch, get_transforms_train_ROSE, get_transforms_valid


class myDataset(Dataset):
    _TRAIN_SPLITS = ("train_manual", "train_sam")
    _EVAL_SPLITS = {"val": ("val",), "test": ("test",)}

    def __init__(self, data_root, target_root, crop_size, data_mode, k_fold=None,
                 imagefile_csv=None, num_fold=None, data_root_aux=None, img_aug=False):
        if data_mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported data_mode: {data_mode}")

        self.crop_size = crop_size
        self.data_root, train_splits = self._resolve_dataset_root(data_root)
        self.target_root = target_root
        self.data_mode = data_mode
        self.img_aug = img_aug
        self.transforms = (
            get_transforms_train_ROSE() if data_mode == "train" else get_transforms_valid()
        )

        split_names = train_splits if data_mode == "train" else self._EVAL_SPLITS[data_mode]
        self.samples = self._collect_samples(split_names)
        self.image_files = [sample[0] for sample in self.samples]

        print(f"{data_mode} dataset: {len(self.samples)}")

    @classmethod
    def _resolve_dataset_root(cls, data_root):
        dataset_root = os.path.abspath(data_root)
        if cls._has_expected_splits(dataset_root):
            return dataset_root, cls._TRAIN_SPLITS

        split_name = os.path.basename(dataset_root)
        if split_name == "image":
            split_name = os.path.basename(os.path.dirname(dataset_root))
            dataset_root = os.path.dirname(os.path.dirname(dataset_root))
        else:
            dataset_root = os.path.dirname(dataset_root)

        if split_name in cls._TRAIN_SPLITS and cls._has_expected_splits(dataset_root):
            return dataset_root, (split_name,)

        raise FileNotFoundError(
            "data_root must be the ROSSA dataset directory or a train_manual/train_sam image directory; "
            f"received: {data_root}"
        )

    @staticmethod
    def _has_expected_splits(dataset_root):
        split_names = ("train_manual", "train_sam", "val", "test")
        return all(
            os.path.isdir(os.path.join(dataset_root, split_name, "image"))
            and os.path.isdir(os.path.join(dataset_root, split_name, "label"))
            for split_name in split_names
        )

    def _collect_samples(self, split_names):
        samples = []
        for split_name in split_names:
            image_dir = os.path.join(self.data_root, split_name, "image")
            label_dir = os.path.join(self.data_root, split_name, "label")
            for file_name in sorted(os.listdir(image_dir), key=self._filename_sort_key):
                image_path = os.path.join(image_dir, file_name)
                if not os.path.isfile(image_path):
                    continue

                label_path = os.path.join(label_dir, file_name)
                if not os.path.isfile(label_path):
                    raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
                samples.append((file_name, image_path, label_path))
        return samples

    @staticmethod
    def _filename_sort_key(file_name):
        stem, _ = os.path.splitext(file_name)
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, image_path, label_path = self.samples[idx]
        image, label = fetch(image_path, label_path)
        if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
            raise ValueError(f"Could not read ROSSA image-label pair: {image_path}, {label_path}")

        if self.data_mode == "train" and self.img_aug:
            augmented = self.transforms(image=image.astype(np.uint8), mask=label.astype(np.uint8))
            image = augmented["image"]
            label = augmented["mask"]

        image, label = convert_to_tensor(image, label)
        label = (label > 0).float().squeeze()

        return {"image": image, "label": label, "file": file_name}