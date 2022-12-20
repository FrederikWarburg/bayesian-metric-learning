import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TrainDataset(Dataset):
    def __init__(self, data_dir):

        dataset_path = os.path.join(data_dir, "CUB_200_2011")

        image_names = np.loadtxt(os.path.join(dataset_path, "images.txt"), dtype=str)
        image_class_labels = np.loadtxt(
            os.path.join(dataset_path, "image_class_labels.txt"), dtype=int
        )
        self.image_path = os.path.join(dataset_path, "images")
        self.labels = image_class_labels[:, 1]
        self.images = image_names[:, 1]

        idx = self.labels <= 100
        self.labels = self.labels[idx]
        self.images = self.images[idx]

        self.transform = transforms.Compose(
            [
                # transforms.RandomRotation(10),
                transforms.RandomResizedCrop(227, scale=(0.08, 1)),
                # transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
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
            Image.open(os.path.join(self.image_path, self.images[idx])).convert("RGB")
        )
        label = self.labels[idx]

        return image, label


class TestDataset(Dataset):
    def __init__(self, data_dir):

        dataset_path = os.path.join(data_dir, "CUB_200_2011")

        image_names = np.loadtxt(os.path.join(dataset_path, "images.txt"), dtype=str)
        image_class_labels = np.loadtxt(
            os.path.join(dataset_path, "image_class_labels.txt"), dtype=int
        )
        self.image_path = os.path.join(dataset_path, "images")

        self.labels = image_class_labels[:, 1]
        self.images = image_names[:, 1]

        idx = self.labels > 100
        self.labels = self.labels[idx]
        self.images = self.images[idx]

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(227),
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
            Image.open(os.path.join(self.image_path, self.images[idx])).convert("RGB")
        )
        label = self.labels[idx]

        return image, label
