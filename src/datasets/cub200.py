import torchvision.datasets as d
from src.data_modules.BaseDataModule import BaseDataModule
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import time
import torch
from PIL import Image


class TrainDataset(Dataset):
    def __init__(self, data_dir, npos=1, nneg=5):

        dataset_path = data_dir / "CUB_200_2011"

        image_names = np.loadtxt(dataset_path / "images.txt", dtype=str)
        image_class_labels = np.loadtxt(
            dataset_path / "image_class_labels.txt", dtype=int
        )
        self.image_path = dataset_path / "images"
        self.labels = image_class_labels[:, 1]
        self.images = image_names[:, 1]

        idx = self.labels <= 100
        self.labels = self.labels[idx]
        self.images = self.images[idx]

        self.npos = npos
        self.nneg = nneg
        self.classes = np.unique(self.labels)

        print("=> this is a bit slow, but only done once")
        t = time.time()

        self.idx = {}
        for c in self.classes:
            self.idx[f"{c}"] = {
                "pos": np.where(self.labels == c)[0],
                "neg": np.where(self.labels != c)[0],
            }

        self.pos_idx = []
        self.neg_idx = []
        for i in range(len(self.labels)):
            key = f"{self.labels[i]}"

            pos_idx = self.idx[key]["pos"]
            pos_idx = pos_idx[pos_idx != i]  # remove self

            neg_idx = self.idx[key]["neg"]

            self.pos_idx.append(pos_idx)
            self.neg_idx.append(neg_idx)
        print("=> done in {:.2f}s".format(time.time() - t))

        self.transform = transforms.Compose(
            [
                transforms.Resize(156),
                transforms.RandomCrop(128),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.labels)

    def load_image(self, path):
        im = Image.open(self.image_path / path).convert("RGB")
        return self.transform(im)

    def __getitem__(self, idx):

        image = self.transform(
            Image.open(self.image_path / self.images[idx]).convert("RGB")
        )
        label = self.labels[idx]

        return image, label


class TestDataset(Dataset):
    def __init__(self, data_dir, npos=1, nneg=5):

        dataset_path = data_dir / "CUB_200_2011"
        self.image_path = dataset_path / "images"
        image_names = np.loadtxt(dataset_path / "images.txt", dtype=str)
        image_class_labels = np.loadtxt(
            dataset_path / "image_class_labels.txt", dtype=int
        )

        self.labels = image_class_labels[:, 1]
        self.images = image_names[:, 1]

        idx = self.labels > 100
        self.labels = self.labels[idx]
        self.images = self.images[idx]

        self.transform = transforms.Compose(
            [
                transforms.Resize(156),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = self.transform(
            Image.open(self.image_path / self.images[idx]).convert("RGB")
        )
        label = self.labels[idx]

        return image, label
