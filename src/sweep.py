import os

import wandb
import yaml
from dotenv import load_dotenv
from dotmap import DotMap

from run import main


def parse_args():

    wandb.init()

    with open(wandb.config.config) as file:
        config = yaml.full_load(file)

    config = DotMap(config)

    if config.data_dir is None:
        config.data_dir = os.getenv("DATA_DIR")

    # defining hard values
    for k, v in wandb.config.items():
        if k != "config":
            # ensure that all values we tune are in the config
            assert k in config

            config[k] = v
            print("==> overwriting ", k, " with ", v)
    print(config)

    wandb.config.seed = 42
    wandb.config.validate_only = False

    return config, wandb.config


if __name__ == "__main__":
    load_dotenv()

    # parse arguments
    config, args = parse_args()
    print(config.margin, config.lr)

    main(config, args, sweep_name="sweep")
