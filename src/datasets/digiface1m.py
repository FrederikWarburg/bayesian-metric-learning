from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import time
from PIL import Image
import os


class TrainDataset(Dataset):
    def __init__(self, data_dir):

        dataset_path = os.path.join(data_dir, "digiface1m")
        image_paths = []
        labels = []
        for label_name in os.listdir(dataset_path):
            try:
                for image_path in os.listdir(os.path.join(dataset_path, label_name) ):
                    image_paths.append(image_path)
                    labels.append(int(label_name))
            except NotADirectoryError:
                pass

        labels = np.array(labels)
        image_paths = np.array(image_paths)
        idx = labels <= 1000 # TODO what split are we going to choose
        self.labels = labels[idx]
        self.images = image_paths[idx]

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(227, scale=(0.08, 1)),
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

        dataset_path = os.path.join(data_dir, "digiface1m")
        image_paths = []
        labels = []
        for label_name in os.listdir(dataset_path):
            try:
                for image_path in os.listdir(os.path.join(dataset_path, label_name) ):
                    image_paths.append(image_path)
                    labels.append(int(label_name))
            except NotADirectoryError:
                pass

        labels = np.array(labels)
        image_paths = np.array(image_paths)
        idx = labels > 1000 # TODO what split are we going to choose
        self.labels = labels[idx]
        self.images = image_paths[idx]

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(227, scale=(0.08, 1)),
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
