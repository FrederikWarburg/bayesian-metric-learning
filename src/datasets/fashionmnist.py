import torch.utils.data as data
import torchvision


class TrainDataset(data.Dataset):
    def __init__(self, root, transform):

        self.data = torchvision.datasets.FashionMNIST(
            root, train=True, download=True, transform=transform
        )

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):

        img, target = self.data.__getitem__(idx)

        return img, target


class TestDataset(data.Dataset):
    def __init__(self, root, transform):
        self.data = torchvision.datasets.FashionMNIST(
            root, train=False, download=True, transform=transform
        )

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):

        img, target = self.data.__getitem__(idx)

        return img, target
