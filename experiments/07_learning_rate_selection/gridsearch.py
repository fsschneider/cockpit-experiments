"""Code for running a grid search."""

import os
import sys

import matplotlib.pyplot as plt
from torch.optim import SGD

from cockpit import quantities

sys.path.append(os.getcwd())
from experiments.utils.deepobs_runner import DeepOBSRunner, fix_deepobs_data_dir  # noqa

# save information
HERE = os.path.abspath(__file__)
SAVEDIR = os.path.join(os.path.dirname(HERE), "results")
DATADIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data_deepobs")
os.makedirs(SAVEDIR, exist_ok=True)


def plot_schedule(global_step):
    """Plotting schedule that does not plot anything."""
    return False


def const_schedule(num_epochs):
    """Constant schedule with a small decay at the end."""
    return lambda epoch: 1.0


def run(problem, lrs, track_schedule):
    """Run a gridsearch for given learning rates tracking the Alpha quantity.

    Args:
        problem (str): String for a DeepOBS test problem, e.g. `"cifar10_3c3d"`.
        lrs ([float]): List of learning rates to test.
        track_schedule (cockpit.utils.schedules): Schedule for tracking the
            Alpha quantity.
    """
    fix_deepobs_data_dir(DATADIR)

    optimizer_class = SGD

    for lr in lrs:
        print("Running: ", lr)
        quants = [quantities.Alpha(track_schedule=track_schedule)]
        hyperparams = {"lr": {"type": float, "default": lr}}
        runner = DeepOBSRunner(
            optimizer_class, hyperparams, quantities=quants, plot_schedule=plot_schedule
        )
        runner.run(
            testproblem=problem,
            l2_reg=0.0,  # necessary for backobs!
            show_plots=False,
            save_plots=False,
            save_final_plot=False,
            save_animation=False,
            lr_schedule=const_schedule,
            output_dir=SAVEDIR,
            skip_if_exists=True,
        )
        plt.close("all")
