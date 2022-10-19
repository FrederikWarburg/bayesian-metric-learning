import wandb
from run import main
import argparse
from dotmap import DotMap
import yaml

# Example sweep configuration
sweep_configuration = {
    "method": "grid",
    "name": "miner_sweep",
    "parameters": {
        "type_of_triplets": {"values" : ["all", "hard", "semihard", "easy"]},
        "max_pairs": {"values" : [1, 10, 50, 100, 500, 1000, 5000]}
    }
}


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="../configs/fashionmnist/laplace_posthoc_fix.yaml",
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
    type_of_triplets = wandb.config.type_of_triplets
    max_pairs = wandb.config.max_pairs

    config, args = parse_args()

    sweep_name = "miner_sweep/type_of_triplet_{}_max_pairs_{}".format(type_of_triplets, max_pairs)
    main(config, args, type_of_triplets=type_of_triplets, max_pairs=max_pairs, sweep_name=sweep_name)


sweep_id = wandb.sweep(sweep_configuration)

# run the sweep
wandb.agent(sweep_id, function=my_train_func)
