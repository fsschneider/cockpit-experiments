"""Benchmark overhead of different cockpit quantities tracking every step."""

import os
import pprint
import sys

from benchmark import benchmark
from utils import (
    get_sys_info,
    settings_baseline,
    settings_configured,
    settings_individual,
)

sys.path.append(os.getcwd())
from experiments.utils.deepobs_runner import fix_deepobs_data_dir  # noqa

# save information
HERE = os.path.abspath(__file__)
SAVEDIR = os.path.join(os.path.dirname(HERE), "results/results_individual")
os.makedirs(SAVEDIR, exist_ok=True)

# general
TRACK_INTERVALS = [1]
NUM_SEEDS = 10
STEPS = 32
HEADER = pprint.pformat(get_sys_info())

CONFIGS = {
    **settings_individual(),
    **settings_baseline(),
    **settings_configured(),
}

# tuple of (problem, device)
RUNS = [
    # GPU
    ("quadratic_deep", "cuda"),
    ("mnist_logreg", "cuda"),
    ("mnist_mlp", "cuda"),
    ("cifar10_3c3d", "cuda"),
    ("fmnist_2c2d", "cuda"),
    # Exact second-order information too expensive
    # ("cifar100_allcnnc", "cuda"),
    # CPU
    ("quadratic_deep", "cpu"),
    ("mnist_logreg", "cpu"),
    ("mnist_mlp", "cpu"),
    ("cifar10_3c3d", "cpu"),
    ("fmnist_2c2d", "cpu"),
    # Exact second-order information too expensive
    # ("cifar100_allcnnc", "cpu"),
]


def get_savefile(tp, dev):
    """Get Savepath based on testproblem and device."""
    return os.path.join(SAVEDIR, f"{tp}_{dev}.csv")


if __name__ == "__main__":
    fix_deepobs_data_dir()

    for tp, dev in RUNS:
        savefile = get_savefile(tp, dev)

        result = benchmark(
            [tp],
            CONFIGS,
            TRACK_INTERVALS,
            NUM_SEEDS,
            [dev],
            steps=STEPS,
            savefile=savefile,
            header=HEADER,
        )

        print(result)
