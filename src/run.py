import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import argparse
from loguru import logger as loguru_logger
import torch
from datetime import datetime
from dotmap import DotMap
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from datasets.placerecognitiondata import PlaceRecognitionDataModule
from datasets.imageretrievaldata import ImageRetrievalDataModule
from lightning.deterministic_model import DeterministicModel

# from lightning.laplace_online_mo
from lightning.pfe_model import PfeModel


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="../configs/cub200/det_model.yaml",
        type=str,
        help="config file",
    )
    parser.add_argument("--seed", default=42, type=int, help="seed")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.full_load(file)

    config = DotMap(config)

    return config, args


def main(config, args, margin=None, lr=None):

    if margin is not None:
        config["margin"] = margin
    if lr is not None:
        config["lr"] = lr

    # reproducibility
    pl.seed_everything(args.seed)

    # slightly different training data modules if we are doing place recognition or
    # more standard image retrieval, where we have descrete labels.
    if config.dataset in ("msls", "dag"):
        data_module = PlaceRecognitionDataModule(**config.toDict())
    elif config.dataset in ("mnist", "fashionmnist", "cub200"):
        data_module = ImageRetrievalDataModule(**config.toDict())

    if config.model == "deterministic":
        model = DeterministicModel(config)
    elif config.model == "pfe":
        model = PfeModel(config)
    elif config.model == "laplace_posthoc":
        model = LaplacePosthocModel(config)
    elif config.model == "laplace_online":
        model = LaplaceOnlineModel(config)
    else:
        raise NotImplementedError

    # setup logger
    savepath = f"../lightning_logs/{config.dataset}/{config.model}"
    logger = WandbLogger(save_dir=f"../logs", name=f"{config.dataset}/{config.model}")

    # lightning trainer
    checkpoint_callback = ModelCheckpoint(
        monitor="val_map/map@5",
        dirpath=f"{savepath}/checkpoints",
        filename="best",  # "{epoch:02d}-{val_map@5:.2f}",
        save_top_k=1,
        mode="max",
        save_last=True,
    )

    # scale learning rate
    config.lr = config.lr * config.batch_size * torch.cuda.device_count()

    callbacks = [LearningRateMonitor(logging_interval="step"), checkpoint_callback]

    # freeze model paramters
    trainer = pl.Trainer.from_argparse_args(
        config,
        accelerator="ddp",
        precision=32,
        max_epochs=config.epochs,
        gpus=torch.cuda.device_count(),
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        progress_bar_refresh_rate=20,
        logger=logger,
        # plugins=DDPPlugin(find_unused_parameters=False),
        callbacks=callbacks,
    )

    loguru_logger.info(f"Start testing!")
    trainer.test(model, datamodule=data_module)

    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)

    loguru_logger.info(f"Start testing!")
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":

    # parse arguments
    config, args = parse_args()
    print(config.margin, config.lr)

    main(config, args)
