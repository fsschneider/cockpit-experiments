"""Run (S)GD on the ``scalar_deep`` problem."""

import os
import sys

import problem
from cockpit.quantities import HessMaxEV, Loss
from cockpit.utils import schedules
from torch.optim import SGD

sys.path.append(os.getcwd())
from experiments.utils.deepobs_runner import DeepOBSRunner  # noqa

problem.register()

optimizer_class = SGD
hyperparams = {
    "lr": {"type": float},
    "momentum": {"type": float, "default": 0.0},
    "nesterov": {"type": bool, "default": False},
}


def lr_schedule(num_epochs):
    """Define a constant learning rate schedule."""
    return lambda epoch: 1.0


track_schedule = schedules.logarithmic(0, 5, steps=200, init=True)
quants = [Loss(track_schedule), HessMaxEV(track_schedule)]

runner = DeepOBSRunner(
    optimizer_class,
    hyperparams,
    quantities=quants,
    plot_schedule=track_schedule,
)
runner.run(
    testproblem="scalar_deep",
    l2_reg=0.0,  # necessary for backobs!
    track_interval=1,  # does not matter
    plot_interval=1,  # does not matter
    show_plots=False,
    save_plots=False,
    save_final_plot=True,
    save_animation=False,
    lr_schedule=lr_schedule,
    skip_if_exists=True,
)
