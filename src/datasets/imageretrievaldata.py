import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms


class ImageRetrievalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: int = 32,
        workers=8,
        training_dataset="mnist",
        **args
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.training_dataset = training_dataset
        self.workers = workers

    def setup(self, stage="fit"):

        transform = transforms.Compose([transforms.ToTensor()])

        if self.training_dataset == "mnist":
            from datasets.mnist import TrainDataset, TestDataset
        elif self.train_dataset == "fashionmnist":
            from datasets.fashionmnist import TrainDataset, TestDataset
        elif self.train_dataset == "cub200":
            from datasets.cub200 import TrainDataset, TestDataset

        if stage == "fit":
            self.train_dataset = TrainDataset(self.data_dir, transform)
            self.val_dataset = TestDataset(self.data_dir, transform)

        elif stage == "test":
            self.test_dataset = TestDataset(self.data_dir, transform)

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
