import argparse
import os

import wandb
import yaml
from dotenv import load_dotenv
from dotmap import DotMap

from run import main

# Example sweep configuration
sweep_configuration = {
    "method": "bayes",
    "name": "margin_lr_sweep",
    "metric": {"name": "val_map/map@5", "goal": "maximize"},
    "parameters": {
        "lr": {"max": 1e-5, "min": 1e-7},
        "margin": {"max": 1.0, "min": 0.0},
    },
}


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/cub200/deterministic.yaml",
        type=str,
        help="config file",
    )
    parser.add_argument("--seed", default=42, type=int, help="seed")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.full_load(file)

    config = DotMap(config)

    if config.data_dir is None:
        config.data_dir = os.getenv("DATA_DIR")

    return config, args


def my_train_func():

    wandb.init()
    margin = wandb.config.margin
    lr = wandb.config.lr

    config, args = parse_args()

    main(
        config,
        args,
        sweep_name="margin_lr_sweep/lr_{}_margin_{}".format(lr, margin),
    )


if __name__ == "__main__":
    load_dotenv()

    sweep_id = wandb.sweep(sweep_configuration)

    # run the sweep
    wandb.agent(sweep_id, function=my_train_func)
