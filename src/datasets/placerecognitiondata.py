import pytorch_lightning as pl
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


def get_transform(image_size, normalize=True):

    # Data loading code
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    w = image_size
    h = int(image_size * 3 / 4) if w not in (224, 448) else image_size
    resize = transforms.Resize((h, w), interpolation=InterpolationMode.BICUBIC)

    transform_list = [resize, transforms.ToTensor()]
    if normalize:
        transform_list.append(norm)

    return transforms.Compose(transform_list)


class PlaceRecognitionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        image_size: int = 640,
        batch_size: int = 32,
        posDistThr=10,
        negDistThr=25,
        train_cities="",
        val_cities="",
        test_cities="",
        workers=8,
        training_dataset="msls",
        **args
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.posDistThr = posDistThr
        self.negDistThr = negDistThr
        self.training_dataset = training_dataset
        self.workers = workers

        if self.training_dataset == "msls":
            self.train_cities = train_cities
            self.val_cities = val_cities
            self.test_cities = test_cities

    def setup(self, stage="fit"):

        transform = get_transform(self.image_size)

        train_input = {
            "name": self.training_dataset,
            "mode": "train",
            "imsize": self.image_size,
            "transform": transform,
            "posDistThr": self.posDistThr,
            "negDistThr": self.negDistThr,
            "root_dir": self.data_dir,
        }

        val_input = {
            "name": self.training_dataset,
            "mode": "val",
            "imsize": self.image_size,
            "transform": transform,
            "posDistThr": self.negDistThr,
            "negDistThr": self.negDistThr,
            "root_dir": self.data_dir,
        }

        test_input = {
            "name": self.training_dataset,
            "mode": "test",
            "imsize": self.image_size,
            "transform": transform,
            "posDistThr": self.negDistThr,
            "negDistThr": self.negDistThr,
            "root_dir": self.data_dir,
        }

        if self.training_dataset == "msls":
            from datasets.msls import TrainDataset, TestDataset

            train_input["cities"] = self.train_cities
            val_input["cities"] = self.val_cities
            test_input["cities"] = self.test_cities

        elif self.training_dataset == "aerialstreet":
            from datasets.aerialstreet import TrainDataset, TestDataset

        if stage == "fit":
            self.train_dataset = TrainDataset(**train_input)
            self.val_dataset = TestDataset(**val_input)

        elif stage == "test":
            self.test_dataset = TestDataset(**test_input)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
        )
