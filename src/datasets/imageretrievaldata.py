import pytorch_lightning as pl
from torch.utils.data import DataLoader


class ImageRetrievalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: int = 32,
        workers=8,
        dataset="mnist",
        **args
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset = dataset
        self.workers = workers

    def setup(self, stage="fit"):

        if self.dataset == "mnist":
            from src.datasets.mnist import TestDataset, TrainDataset
        elif self.dataset == "fashionmnist":
            from src.datasets.fashionmnist import TestDataset, TrainDataset
            from src.datasets.mnist import TestDataset as OODDataset
        elif self.dataset == "cub200":
            from src.datasets.cars196 import TestDataset as OODDataset
            from src.datasets.cub200 import TestDataset, TrainDataset
        elif self.dataset == "lfw":
            from src.datasets.lfw import TestDataset, TrainDataset

        if stage == "fit":
            self.train_dataset = TrainDataset(self.data_dir)
            self.val_dataset = TestDataset(self.data_dir)

        elif stage == "test":
            self.test_dataset = TestDataset(self.data_dir)

        if self.dataset in ("fashionmnist", "cub200"):
            self.ood_dataset = OODDataset(self.data_dir)

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
        dataloaders = [
            DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                drop_last=False,
            )
        ]

        if hasattr(self, "ood_dataset"):
            dataloaders += [
                DataLoader(
                    self.ood_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.workers,
                    pin_memory=True,
                    drop_last=False,
                )
            ]

        return dataloaders

    def test_dataloader(self):
        dataloaders = [
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                drop_last=False,
            )
        ]

        if hasattr(self, "ood_dataset"):
            dataloaders += [
                DataLoader(
                    self.ood_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.workers,
                    pin_memory=True,
                    drop_last=False,
                )
            ]

        return dataloaders
