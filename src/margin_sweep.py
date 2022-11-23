import wandb
from run import main
import argparse
from dotmap import DotMap
import yaml

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
        default="../configs/cub200/deterministic.yaml",
        type=str,
        help="config file",
    )
    parser.add_argument("--seed", default=42, type=int, help="seed")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.full_load(file)

    config = DotMap(config)

    return config, args


def my_train_func():

    wandb.init()
    margin = wandb.config.margin
    lr = wandb.config.lr

    config, args = parse_args()

    main(
        config,
        args,
        margin=margin,
        lr=lr,
        sweep_name="margin_lr_sweep/lr_{}_margin_{}".format(lr, margin),
    )


sweep_id = wandb.sweep(sweep_configuration)

# run the sweep
wandb.agent(sweep_id, function=my_train_func)
