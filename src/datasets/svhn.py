import torch.utils.data as data
import torchvision
from torchvision import transforms


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

class TrainDataset(data.Dataset):
    def __init__(self, root):

        self.data = torchvision.datasets.SVHN(
            root, train=True, download=True, transform=transform
        )

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):

        img, target = self.data.__getitem__(idx)

        return img, target


class TestDataset(data.Dataset):
    def __init__(self, root):

        self.data = torchvision.datasets.SVHN(
            root, split="test", download=True, transform=transform
        )

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):

        img, target = self.data.__getitem__(idx)

        return img, target
