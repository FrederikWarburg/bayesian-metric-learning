from typing import Tuple

import numpy as np
import torch.utils.data as data
import torchvision
from PIL import Image
from torch import Tensor
from torchvision import transforms


class TrainDataset(data.Dataset):
    def __init__(self, root):
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
        data = torchvision.datasets.LFWPeople(root, split="train", download=True)

        self.targets = np.array(data.targets)
        self.image_paths = data.data

        unique = {}
        for class_idx in np.unique(self.targets):
            # find the idx for each class
            # {class_idx: pos_idx}
            unique[str(class_idx)] = np.where(self.targets == class_idx)[0][0]

        self.pidxs = []
        self.qidx = []
        for idx, class_idx in enumerate(self.targets):
            # for each position, list all the targets with the same class exept ifself
            pos_idx = unique[str(class_idx)]
            pos_idx = pos_idx[pos_idx != idx]

            # at least one of the same class
            if len(pos_idx) > 0:
                self.pidxs.append(pos_idx)
                self.qidx.append(idx)

    def __len__(self) -> int:
        return len(self.qidx)

    def __getitem__(self, idx) -> Tuple[Tensor, int]:

        qidx = self.qidx[idx]
        pidxs = self.pidxs[idx]

        pidx = np.random.choice(pidxs, 1)[0]

        qimg = Image.open(self.image_paths[qidx])
        pimg = Image.open(self.image_paths[pidx])

        output = [qimg, pimg]
        output = [self.transform(img) for img in output]

        assert self.targets[qidx] == self.targets[pidx]
        target = [self.targets[qidx], self.targets[pidx]]

        return output, target

    # def __getitem__(self, idx) -> Tuple[Tensor, Tensor, int]:
    #     img1, img2, target = self.data.__getitem__(idx)
    #     return img1, img2, target


class TestDataset(data.Dataset):
    def __init__(self, root):
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(227),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

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
