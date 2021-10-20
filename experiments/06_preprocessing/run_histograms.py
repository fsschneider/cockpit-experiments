"""Compute the 2d histogram for different data pre-processing steps."""

import glob
import os
import sys

from run_samples import make_and_register_tproblems
from torch.optim import SGD

from cockpit import quantities

sys.path.append(os.getcwd())
from experiments.utils.deepobs_runner import DeepOBSRunner, fix_deepobs_data_dir  # noqa

fix_deepobs_data_dir()

optimizer_class = SGD
hyperparams = {"lr": {"type": float, "default": 0.001}}

HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)
SAVEDIR = os.path.join(HEREDIR, "results")
os.makedirs(SAVEDIR, exist_ok=True)


def lr_schedule(num_epochs):
    """Some Learning rate schedule."""
    return lambda epoch: 0.0


def plot_schedule(global_step):
    """Plot schedule for only the very first step."""
    return global_step == 0


def make_quantities():
    """Return the 2d Histogram."""

    def track_schedule(global_step):
        return global_step == 0

    quants = [
        quantities.GradHist2d(
            range=((-1.5, 1.5), (-0.2, 0.2)),
            track_schedule=track_schedule,
        )
    ]

    return quants


def get_out_file(tproblem):
    """Return the filepath to logfiles."""
    probpath = os.path.join(HEREDIR, "results", tproblem, "SGD")
    pattern = os.path.join(probpath, "*", "*__log.json")

    filepath = glob.glob(pattern)
    if len(filepath) != 1:
        raise ValueError(f"Found no or multiple files: {filepath}, pattern: {pattern}")
    return filepath[0]


make_and_register_tproblems()

PROBLEMS = ["cifar10raw_3c3d", "cifar10scale255_3c3d"]

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
