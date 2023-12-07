import argparse
from typing import Tuple

from dotmap import DotMap

import yaml

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
    args = parser.parse_args()
    config_path = args.config

    with open(config_path) as file:
        config = yaml.full_load(file)

    config = DotMap(config)
    print(config)

    return config, args


if __name__ == '__main__':

    run_training(parse_args("../configs/fashionmnist/deterministic.yaml", 1))
    run_training(parse_args("../configs/fashionmnist/deterministic.yaml", 2))
    run_training(parse_args("../configs/fashionmnist/deterministic.yaml", 3))
    run_training(parse_args("../configs/fashionmnist/deterministic.yaml", 4))
    run_training(parse_args("../configs/fashionmnist/deterministic.yaml", 5))

    run_training(parse_args("../configs/fashionmnist/laplace_posthoc_arccos_fix.yaml", 42))
    run_training(parse_args("../configs/fashionmnist/laplace_online_arccos_fix.yaml", 42))

    run_training(parse_args("../configs/fashionmnist/mcdrop.yaml", 42))
    run_training(parse_args("../configs/fashionmnist/pfe.yaml", 42))

    run_training(parse_args("../configs/fashionmnist/deepensemble.yaml", 42))

