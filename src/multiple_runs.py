import argparse
from typing import Tuple

import wandb
import yaml
from dotmap import DotMap

from run import main as run_training


def parse_args(cfg_path: str, seed: int) -> Tuple[DotMap, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=cfg_path,
        type=str,
        help="config file",
    )
    parser.add_argument("--seed", default=seed, type=int, help="seed")
    parser.add_argument("--validate-only", action='store_true', help="only validate")
    parser.add_argument("--resume_from_checkpoint", default=None, type=str, help="resume from checkpoint")
    parser.add_argument("--data_type", type=str, default="fashionmnist") # Needs to be here to not break everything
    args = parser.parse_args()
    config_path = args.config

    with open(config_path) as file:
        config = yaml.full_load(file)

    config = DotMap(config)
    print(config)

    return config, args


def parse_data_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="fashionmnist")
    arg = parser.parse_args()

    return arg.data_type

if __name__ == '__main__':
    dtype = parse_data_arg()

    run_training(*parse_args(f"../configs/{dtype}/deterministic.yaml", 1))
    wandb.finish()
    run_training(*parse_args(f"../configs/{dtype}/deterministic.yaml", 2))
    wandb.finish()
    run_training(*parse_args(f"../configs/{dtype}/deterministic.yaml", 3))
    wandb.finish()
    run_training(*parse_args(f"../configs/{dtype}/deterministic.yaml", 4))
    wandb.finish()
    run_training(*parse_args(f"../configs/{dtype}/deterministic.yaml", 5))
    wandb.finish()
    run_training(*parse_args(f"../configs/{dtype}/deterministic.yaml", 42))
    wandb.finish()

    run_training(*parse_args(f"../configs/{dtype}/laplace_posthoc_arccos_fix.yaml", 42))
    wandb.finish()
    run_training(*parse_args(f"../configs/{dtype}/laplace_online_arccos_fix.yaml", 42))
    wandb.finish()

    run_training(*parse_args(f"../configs/{dtype}/mcdrop.yaml", 42))
    wandb.finish()
    run_training(*parse_args(f"../configs/{dtype}/pfe.yaml", 42))
    wandb.finish()

    run_training(*parse_args(f"../configs/{dtype}/deepensemble.yaml", 42))
    wandb.finish()
