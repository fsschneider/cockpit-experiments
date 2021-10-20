"""Compute the 2d histogram for different data pre-processing steps."""

import os
import sys

from torch.optim import SGD

from cockpit import quantities

sys.path.append(os.getcwd())
from experiments.utils.custom_cifar10_3c3d import (  # noqa
    cifar10_3c3dsig,
    make_cifar10transform_3c3d,
    make_cifar10transform_3c3dsig,
)
from experiments.utils.deepobs_runner import (  # noqa
    DeepOBSRunner,
    fix_deepobs_data_dir,
    register,
)

register(cifar10_3c3dsig)

HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)
SAVEDIR = os.path.join(HEREDIR, "results")
os.makedirs(SAVEDIR, exist_ok=True)

fix_deepobs_data_dir()

optimizer_class = SGD
hyperparams = {"lr": {"type": float, "default": 0.001}}


def lr_schedule(num_epochs):
    """Some Learning rate schedule."""
    return lambda epoch: 0.0


def plot_schedule(global_step):
    """Plot schedule for only the first step."""
    return global_step == 0


def make_quantities():
    """Track quantities only in first step, use the histogram."""

    def track_schedule(global_step):
        return global_step == 0

    return [
        quantities.GradHist2d(
            range=((-1.1, 1.1), (-0.2, 0.2)),
            track_schedule=track_schedule,
            keep_individual=True,
            bins=(20, 25),
        )
    ]


def scale255(tensor):
    """Scale tensor by 255."""
    return 255.0 * tensor


TRANSFORMS = {}
PROBLEMS = []

# build and register testproblems with ReLU
PROBLEMS += ["cifar10_3c3d"]

for trafo_name, trafo in TRANSFORMS.items():
    make_cifar10transform_3c3d(trafo, trafo_name)

PROBLEMS += [f"cifar10{trafo_name}_3c3d" for trafo_name in TRANSFORMS.keys()]

# build and register testproblems with Sigmoids
PROBLEMS += ["cifar10_3c3dsig"]

for trafo_name, trafo in TRANSFORMS.items():
    make_cifar10transform_3c3dsig(trafo, trafo_name)

PROBLEMS += [f"cifar10{trafo_name}_3c3dsig" for trafo_name in TRANSFORMS.keys()]

if __name__ == "__main__":

    for problem in PROBLEMS:
        runner = DeepOBSRunner(
            optimizer_class,
            hyperparams,
            quantities=make_quantities(),
            plot_schedule=plot_schedule,
        )

        runner.run(
            testproblem=problem,
            l2_reg=0.0,  # necessary for backobs!
            num_epochs=1,
            batch_size=128,
            show_plots=False,
            save_plots=True,
            save_final_plot=False,
            save_animation=False,
            lr_schedule=lr_schedule,
            skip_if_exists=True,
            output_dir=SAVEDIR,
        )
