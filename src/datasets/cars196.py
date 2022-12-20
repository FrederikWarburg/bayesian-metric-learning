import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TrainDataset(Dataset):
    def __init__(self, data_dir):

        dataset_path = os.path.join(data_dir, "cars196")
        self.image_path = dataset_path
        data_train = np.loadtxt(os.path.join(dataset_path, "anno_train.csv"), delimiter=",", dtype=str)
        names_train = [f"cars_train/{x}" for x in data_train[:, 0]]

        data_test = np.loadtxt(os.path.join(dataset_path, "anno_test.csv"), delimiter=",", dtype=str)
        names_test = [f"cars_test/{x}" for x in data_test[:, 0]]

        data = np.concatenate((data_train, data_test), axis=0)

        self.labels = data[:, -1].astype(int)
        self.images = np.concatenate((names_train, names_test), axis=0)

        idx = self.labels < 98
        self.labels = self.labels[idx]
        self.images = self.images[idx]
        self.transform = transforms.Compose(
            [
                # transforms.RandomRotation(10),
                transforms.RandomResizedCrop(227, scale=(0.08, 1)),
                # transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = self.transform(Image.open(os.path.join(self.image_path, self.images[idx])).convert("RGB"))
        label = self.labels[idx]

        return image, label


class TestDataset(Dataset):
    def __init__(self, data_dir):

        dataset_path = os.path.join(data_dir, "cars196")
        self.image_path = dataset_path
        data_train = np.loadtxt(os.path.join(dataset_path, "anno_train.csv"), delimiter=",", dtype=str)
        names_train = [f"cars_train/{x}" for x in data_train[:, 0]]

        data_test = np.loadtxt(os.path.join(dataset_path, "anno_test.csv"), delimiter=",", dtype=str)
        names_test = [f"cars_test/{x}" for x in data_test[:, 0]]

        data = np.concatenate((data_train, data_test), axis=0)

        self.labels = data[:, -1].astype(int)
        self.images = np.concatenate((names_train, names_test), axis=0)

        idx = self.labels > 98
        self.labels = self.labels[idx]
        self.images = self.images[idx]

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(227),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = self.transform(Image.open(os.path.join(self.image_path, self.images[idx])).convert("RGB"))
        label = self.labels[idx]

        return image, label
