from typing import Tuple

import torch.utils.data as data
import torchvision
from torch import Tensor
from torchvision import transforms


class TrainDataset(data.Dataset):
    def __init__(self, root):
        transform = transforms.Compose([
            # transforms.RandomResizedCrop(227, scale=(0.08, 1)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # ),
        ])
        self.data = torchvision.datasets.LFWPeople(
            root, split="train", download=True, transform=transform
        )

    def __len__(self) -> int:
        return self.data.__len__()

    def __getitem__(self, idx) -> Tuple[Tensor, int]:
        img, target = self.data.__getitem__(idx)
        return img, target

    # def __getitem__(self, idx) -> Tuple[Tensor, Tensor, int]:
    #     img1, img2, target = self.data.__getitem__(idx)
    #     return img1, img2, target


class TestDataset(data.Dataset):
    def __init__(self, root):
        transform = transforms.Compose([
            # transforms.RandomResizedCrop(227, scale=(0.08, 1)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # ),
        ])
        self.data = torchvision.datasets.LFWPeople(
            root, split="test", download=True, transform=transform
        )

    def __len__(self) -> int:
        return self.data.__len__()

    def __getitem__(self, idx) -> Tuple[Tensor, int]:
        img, target = self.data.__getitem__(idx)
        return img, target

    # def __getitem__(self, idx) -> Tuple[Tensor, Tensor, int]:
    #     img1, img2, target = self.data.__getitem__(idx)
    #     return img1, img2, target
