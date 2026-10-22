"""So2Sat LCZ42 (GeoBench v2): Sentinel-1 SAR + Sentinel-2 optical.

Local Climate Zone classification over 17 balanced classes, read from the
per-split `.npy` arrays written by `scripts/convert_so2sat.py`. The published
distribution is a single `.tortilla`, whose reader needs Python >= 3.9 while the
training environment is 3.8; converting once keeps this module dependency-free
and memory-mappable, the same route CREMA-D's features take.

The two modalities are the two sensors, so this is the classification counterpart
of Sen1Floods11's segmentation and uses the same encoder family.
"""
import json
import os
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

MODALITIES = ("s1", "s2")


class So2SatBase(Dataset):
    """One split, with both sensors memory-mapped.

    Patches are 32x32 and `image_size` defaults to that, so the resize is a no-op
    and the ViT encoders tokenise the native grid. Larger values still work, for
    an encoder that needs them.
    """

    def __init__(self, root: str, split: str = "train", image_size: int = 224,
                 normalizers=None):
        self.root, self.split = root, split
        self.data = {m: np.load(os.path.join(root, f"{split}_{m}.npy"), mmap_mode="r")
                     for m in MODALITIES}
        self.labels = np.load(os.path.join(root, f"{split}_labels.npy"))
        self.normalizers = normalizers
        self.resize = v2.Resize(image_size, antialias=True)

    def __len__(self):
        return len(self.labels)

    def get_raw_item(self, idx):
        """Normalise at 32x32, where the statistics were measured, then resize.

        Normalising after the resize would apply them to a distribution the
        interpolation has already smoothed.
        """
        out = []
        for i, m in enumerate(MODALITIES):
            x = torch.from_numpy(np.array(self.data[m][idx], dtype=np.float32))
            if self.normalizers is not None:
                x = self.normalizers[i](x)
            out.append(self.resize(x))
        return out, torch.tensor(int(self.labels[idx]), dtype=torch.long)


class So2SatSup(So2SatBase):
    """Supervised view, for the linear probe: `([s1, s2], label)`."""

    def __init__(self, root, split="train", image_size=224,
                 spatial_transform=None, normalizers=None):
        super().__init__(root, split=split, image_size=image_size,
                         normalizers=normalizers)
        self.spatial_transform = spatial_transform

    def __getitem__(self, idx):
        mods, label = self.get_raw_item(idx)
        if self.spatial_transform is not None:
            mods = list(self.spatial_transform(*mods))
        return mods, label


class So2SatMMSSL(So2SatBase):
    """Two augmented views for the self-supervised objectives.

    Returns `(view1, view2)` and no label: `BaseModel.training_step` calls
    `forward(*batch)`, so a third element silently shifts the views into the
    wrong arguments instead of raising.
    """

    def __init__(self, root, split="train", image_size=224,
                 spatial_transform=None, pixel_transform=None, normalizers=None):
        super().__init__(root, split=split, image_size=image_size,
                         normalizers=normalizers)
        self.spatial_transform = spatial_transform
        self.pixel_transform = pixel_transform

    def _one_view(self, mods):
        if self.spatial_transform is not None:
            mods = list(self.spatial_transform(*mods))
        if self.pixel_transform is not None:
            mods = [self.pixel_transform(x) for x in mods]
        return mods

    def __getitem__(self, idx):
        mods, _ = self.get_raw_item(idx)
        return self._one_view(mods), self._one_view(mods)


class So2SatDataModule(LightningDataModule):
    """So2Sat for pre-training (`model` in {CoMM, WoMM, MMSD}) or probing (`Sup`).

    The validation split doubles as the test split for the probe, as on the other
    datasets: nothing is selected on it during pre-training and the linear probe
    is the reported measurement.
    """

    def __init__(self, model: str,
                 image_size: int = 224,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 **kwargs):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        catalog_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "catalog.json")
        with open(catalog_path) as f:
            self.root = json.load(f)["so2sat"]["path"]
        with open(os.path.join(self.root, "meta.json")) as f:
            meta = json.load(f)
        self.classes = meta["classes"]

        # Statistics measured on the training split by the converter rather than
        # copied. They are applied positionally, so they hold regardless of which
        # of GeoBench's two conflicting `normalization_stats` definitions is the
        # intended one.
        self.normalizers = [v2.Normalize(mean=meta["norm"][m]["mean"],
                                         std=meta["norm"][m]["std"]) for m in MODALITIES]

        self.spatial_augment = v2.Compose([
            v2.RandomResizedCrop(image_size, scale=(0.5, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
        ])
        # The blur kernel is a fraction of the image, not a constant: 11 px was
        # chosen against a 224 px input (~5%), and reusing it on a 32 px patch
        # would erase the patch rather than blur it.
        k = max(3, int(round(0.05 * image_size)) | 1)
        self.pixel_augment = v2.Compose([
            v2.RandomApply([v2.GaussianBlur(kernel_size=(k, k))], p=0.8),
        ])

        # The probe callbacks hold a data module that is never handed to a Trainer,
        # so Lightning never calls setup() on it. Build here, as Sen1Floods11 does.
        self.setup()

    def setup(self, stage=None):
        if getattr(self, "train_dataset", None) is not None:
            return
        if self.model in ("CoMM", "WoMM", "MMSD"):
            common = dict(root=self.root, image_size=self.image_size,
                          spatial_transform=self.spatial_augment,
                          pixel_transform=self.pixel_augment,
                          normalizers=self.normalizers)
            self.train_dataset = So2SatMMSSL(split="train", **common)
            self.val_dataset = So2SatMMSSL(split="validation", **common)
            self.test_dataset = So2SatMMSSL(split="test", **common)
        elif self.model == "Sup":
            common = dict(root=self.root, image_size=self.image_size,
                          normalizers=self.normalizers)
            self.train_dataset = So2SatSup(split="train",
                                           spatial_transform=self.spatial_augment, **common)
            # The linear probe selects its cost on validation and refits on train and
            # validation together, so the reported split must be a third one.
            self.val_dataset = So2SatSup(split="validation", **common)
            self.test_dataset = So2SatSup(split="test", **common)
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def _loader(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          drop_last=shuffle)

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, False)
