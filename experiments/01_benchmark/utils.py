"""Utility functions for benchmarks."""

import os
import subprocess
import sys
from functools import lru_cache

from deepobs.pytorch.testproblems import (
    cifar10_3c3d,
    cifar100_3c3d,
    cifar100_allcnnc,
    fmnist_2c2d,
    fmnist_mlp,
    mnist_2c2d,
    mnist_logreg,
    mnist_mlp,
    quadratic_deep,
)
from pytest_benchmark.plugin import (
    get_commit_info,
    pytest_benchmark_generate_machine_info,
)

from cockpit.quantities import Time
from cockpit.utils.configuration import quantities_cls_for_configuration

sys.path.append(os.getcwd())
from experiments.utils.custom_imagenet_resnet import dummyimagenet_resnet50nobn  # noqa


def settings_individual():
    """Return benchmark quantities to measure overhead of individual quantities."""
    configs = {}

    for q in quantities_cls_for_configuration("full"):
        if not q == Time:
            configs[q.__name__] = [q, Time]

    return configs


def settings_configured():
    """Return all cockpit configurations."""
    configs = {}

    for label in ["economy", "business", "full"]:
        configs[label] = quantities_cls_for_configuration(label)
        configs[label].append(Time)

    return configs


def settings_baseline():
    """Return baseline with Time quantity."""
    return {"baseline": [Time]}


def get_sys_info():
    """Return system information for benchmark."""
    machine_info = pytest_benchmark_generate_machine_info()
    machine_info.pop("node")

    info = {
        "machine": machine_info,
        "commit": get_commit_info(),
    }
    try:
        info["gpu"] = _get_gpu_info()
    except Exception:
        info["gpu"] = "Unknown"

    return info


def _get_gpu_info(keys=("Product Name", "CUDA Version")):
    """Parse output of nvidia-smi into a python dictionary.

    Link:
        - https://gist.github.com/telegraphic/ecb8161aedb02d3a09e39f9585e91735

    Args:
        keys (tuple, optional): Keys that should be extracted.
            Defaults to ("Product Name", "CUDA Version").

    Returns:
        [dict]: Dictionary holding the GPU info.
    """
    sp = subprocess.Popen(
        ["nvidia-smi", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out_list = sp.communicate()[0].decode("utf-8").split("\n")

    info = {}

    for item in out_list:
        try:
            key, val = item.split(":")
            key, val = key.strip(), val.strip()
            if key in keys:
                info[key] = val
        except Exception:
            pass

    return info


@lru_cache()
def get_train_size(testproblem):
    """Return number of samples in training set."""
    tproblem_cls_from_str = {
        "cifar10_3c3d": cifar10_3c3d,
        "cifar100_3c3d": cifar100_3c3d,
        "cifar100_allcnnc": cifar100_allcnnc,
        "fmnist_2c2d": fmnist_2c2d,
        "fmnist_mlp": fmnist_mlp,
        "mnist_2c2d": mnist_2c2d,
        "mnist_logreg": mnist_logreg,
        "mnist_mlp": mnist_mlp,
        "quadratic_deep": quadratic_deep,
        "dummyimagenet_resnet50nobn": dummyimagenet_resnet50nobn,
    }
    tproblem_cls = tproblem_cls_from_str[testproblem]

    return _get_train_size(tproblem_cls)


def _get_train_size(tproblem_cls):
    """Return number of samples in training set."""
    batch_size = 1

    tproblem = tproblem_cls(batch_size=batch_size)
    tproblem.set_up()

    return _get_train_steps_per_epoch(tproblem) * batch_size


def _get_train_steps_per_epoch(tproblem):
    """Return number of mini-batches in the train set."""
    tproblem.train_init_op()

    steps = 0

    try:
        while True:
            tproblem._get_next_batch()
            steps += 1
    except StopIteration:
        return steps
    except Exception as e:
        raise RuntimeError(f"Failed to detect steps per epoch: {e}")
