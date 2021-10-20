"""Run SGD on MNIST logistic regression with a cyclic LR schedule."""

import os
import sys
from functools import partial

from shared import (
    animate,
    cosine_decay_restarts_schedule,
    hyperparams,
    optimizer_class,
    plot_schedule,
    plot_to_tex_dir,
    track_schedule,
)

from cockpit import quantities
from cockpit.utils.configuration import quantities_cls_for_configuration

sys.path.append(os.getcwd())
from experiments.utils.deepobs_runner import DeepOBSRunner, fix_deepobs_data_dir  # noqa

fix_deepobs_data_dir()

# save information
HERE = os.path.abspath(__file__)
SAVEDIR = os.path.join(os.path.dirname(HERE), "results")
DATADIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data_deepobs")
os.makedirs(SAVEDIR, exist_ok=True)

# lr schedule
steps_for_cycle = 4
lr_schedule = partial(cosine_decay_restarts_schedule, steps_for_cycle)


# quantities
quants = []
for q in quantities_cls_for_configuration("full"):
    if q == quantities.GradHist2d:
        quants.append(
            q(
                # mnist_logreg is initialized with zeros, need to set limits manually
                range=((-1, 1), (-0.5, 0.5)),
                track_schedule=track_schedule,
                verbose=True,
            )
        )
    else:
        quants.append(q(track_schedule=track_schedule, verbose=True))

# run
runner = DeepOBSRunner(
    optimizer_class, hyperparams, quantities=quants, plot_schedule=plot_schedule
)

testproblem = "mnist_logreg"
runner.run(
    testproblem=testproblem,
    l2_reg=0.0,  # necessary for backobs!
    show_plots=False,
    save_plots=True,
    save_final_plot=True,
    save_animation=True,
    lr_schedule=lr_schedule,
    output_dir=SAVEDIR,
    skip_if_exists=True,
)

# visualize
plot_to_tex_dir(testproblem, optimizer_class)
animate(testproblem, optimizer_class)
