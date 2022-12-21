import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import argparse
from loguru import logger as loguru_logger
import torch
from datetime import datetime
from dotmap import DotMap
import yaml
import os
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import sys

# data modules
from datasets.placerecognitiondata import PlaceRecognitionDataModule
from datasets.imageretrievaldata import ImageRetrievalDataModule

# models
from lightning.deterministic_model import DeterministicModel
from lightning.laplace_posthoc_model import LaplacePosthocModel
from lightning.laplace_online_model import LaplaceOnlineModel
from lightning.pfe_model import PfeModel
from lightning.mcdropout_model import MCDropoutModel
from lightning.deep_ensemble_model import DeepEnsembleModel
from lightning.hib_model import HibModel

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="../configs/cub200/det_model.yaml",
        type=str,
        help="config file",
    )
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--validate-only", action='store_true', help="only validate")
    args = parser.parse_args()
    config_path = args.config

    with open(config_path) as file:
        config = yaml.full_load(file)

    config = DotMap(config)
    print(config)

    return config, args


models = {
    "deterministic": DeterministicModel,
    "pfe": PfeModel,
    "laplace_online": LaplaceOnlineModel,
    "laplace_posthoc": LaplacePosthocModel,
    "mc_dropout": MCDropoutModel,
    "deep_ensemble": DeepEnsembleModel,
    "hib": HibModel,
}


def main(
    config,
    args,
    sweep_name="",
):

    # reproducibility
    pl.seed_everything(args.seed)

    # slightly different training data modules if we are doing place recognition or
    # more standard image retrieval, where we have descrete labels.
    if config.dataset in ("msls", "dag"):
        data_module = PlaceRecognitionDataModule(**config.toDict())
    elif config.dataset in ("mnist", "fashionmnist", "cub200", "lfw", "cifar10"):
        data_module = ImageRetrievalDataModule(**config.toDict())
    else:
        raise NotImplementedError(f"Dataset {config.dataset} not implemented")
    data_module.setup()
    config["dataset_size"] = data_module.train_dataset.__len__()

    name = f"{sweep_name}{config.dataset}/{config.model}/{args.seed}"
    if "laplace" in config.model:
        name += f"/{config.loss}"
        name += f"/{config.loss_approx}"
    savepath = f"../lightning_logs/{name}"

    model = models[config.model](config, savepath=savepath, seed=args.seed)

    # setup logger
    os.makedirs("../logs", exist_ok=True)
    logger = WandbLogger(save_dir=f"../logs", name=name)

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

    if "sweep" in sweep_name:
        callbacks = [LearningRateMonitor(logging_interval="step")]
    else:
        callbacks = [LearningRateMonitor(logging_interval="step"), checkpoint_callback]

    # freeze model paramters
    trainer = pl.Trainer.from_argparse_args(
        config,
        accelerator="gpu",
        precision=32,
        max_epochs=config.epochs,
        devices=torch.cuda.device_count(),
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        logger=logger,
        # plugins=DDPPlugin(find_unused_parameters=False),
        callbacks=callbacks,
    )

    if args.validate_only:
        model_path = os.path.join(savepath, "checkpoints", "best.ckpt")
        
        if os.path.isfile(model_path):
            statedict = torch.load(model_path)
            statedict = statedict["state_dict"] if "state_dict" in statedict else statedict
            model.load_state_dict(statedict)
        else:
            print("checkpoint not found at ")
            print(model_path)
            sys.exit()
    else:
        if config.model in ("laplace_posthoc"):
            loguru_logger.info(f"Start training!")
            model.fit(datamodule=data_module)
        elif config.model in ("deep_ensemble"):
            pass
        else:
            loguru_logger.info(f"Start testing!")
            # trainer.test(model, datamodule=data_module)

            loguru_logger.info(f"Start training!")
            trainer.fit(model, datamodule=data_module)

    if config.get("optimize_prior_prec", False):
        print("==> optimize prior precision")
        prior_prec = config.get("prior_prec", 1)
        if prior_prec == 1:
            model.optimize_prior_precision()
        else:
            model.prior_prec = prior_prec
    
    loguru_logger.info(f"Start testing!")
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":

    # parse arguments
    config, args = parse_args()
    main(config, args)
