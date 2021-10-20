"""Run SGD on the local quadratic problem."""

import os
import sys

from torch.optim import SGD

import cockpit.utils.schedules as schedules
from cockpit import quantities

sys.path.append(os.getcwd())
from experiments.utils.deepobs_runner import DeepOBSRunner  # noqa
from experiments.utils.two_d_quadratic import register  # noqa

# save information
HERE = os.path.abspath(__file__)
SAVEDIR = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(SAVEDIR, exist_ok=True)

register()

optimizer_class = SGD
hyperparams = {
    "lr": {"type": float, "default": 1.0},
    "momentum": {"type": float, "default": 0.0},
    "nesterov": {"type": bool, "default": False},
}


# schedules
def lr_schedule(num_epochs):
    """Some Learning rate schedule."""
    return lambda epoch: 0.2


track_schedule = schedules.linear(1)
plot_schedule = track_schedule


# quantities
def make_quantities():
    """Create a list of quantities."""
    quants = []
    for q_cls in [
        quantities.Parameters,
        quantities.Alpha,
        quantities.Loss,
        quantities.Distance,
        quantities.GradNorm,
        quantities.Time,
    ]:
        quants.append(q_cls(track_schedule=track_schedule, verbose=False))

    return quants


runner = DeepOBSRunner(
    optimizer_class,
    hyperparams,
    quantities=make_quantities(),
    plot_schedule=plot_schedule,
)
runner.run(
    testproblem="two_d_quadratic",
    l2_reg=0.0,  # necessary for backobs!
    num_epochs=20,
    batch_size=128,
    show_plots=False,
    save_plots=False,
    save_final_plot=False,
    save_animation=False,
    lr_schedule=lr_schedule,
    output_dir=SAVEDIR,
)


def lr_schedule_decay(num_epochs):
    """A manually tuned schedule to emulate the loss curve of the other run."""
    r = 0.87
    a = 7.2 / 1000
    return lambda epoch: a * r ** epoch


runner = DeepOBSRunner(
    optimizer_class,
    hyperparams,
    quantities=make_quantities(),
    plot_schedule=plot_schedule,
)

runner.run(
    testproblem="two_d_quadratic",
    l2_reg=0.0,  # necessary for backobs!
    num_epochs=20,
    batch_size=128,
    show_plots=False,
    save_plots=False,
    save_final_plot=False,
    save_animation=False,
    lr_schedule=lr_schedule_decay,
    output_dir=SAVEDIR,
)
