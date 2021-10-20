"""Benchmark overhead of quantities and tracking intervals on large-scale problems.

Memory consumption tends to increase as the benchmark proceeds and it might be killed
by the OS. In this case, simply restart the script and it will continue from where it
was stopped. You may want to invoke the script in an infinite loop to automatically
trigger such a restart:

   ``while true; do python run_grid_imagenet.py; done``

Note that you need to escape from the loop yourself with ``C-c C-c`` to stop it.
"""

import os
import pprint
import sys

from benchmark import benchmark
from run_grid import get_savefile
from utils import get_sys_info, settings_baseline, settings_configured

sys.path.append(os.getcwd())
from experiments.utils.custom_imagenet_resnet import register_imagenet_problem  # noqa
from experiments.utils.deepobs_runner import fix_deepobs_data_dir  # noqa

# switch between minimal (short run time, for debugging) and final settings
USE_MINIMAL = False

# general
HEADER = pprint.pformat(get_sys_info())

if USE_MINIMAL:
    TRACK_INTERVALS = [1, 4]
    NUM_SEEDS = 2
    STEPS = 8
else:
    TRACK_INTERVALS = [1, 4, 16, 64, 256]
    NUM_SEEDS = 10
    STEPS = 512

CONFIGS = {
    **settings_baseline(),
    **settings_configured(),
}
# remove unsupported settings that contain second-order extensions
CONFIGS.pop("business")
CONFIGS.pop("full")


# tuple of (problem, device)
RUNS = [
    ("dummyimagenet_resnet50nobn", "cuda"),
]


if __name__ == "__main__":
    fix_deepobs_data_dir()
    register_imagenet_problem()

    # run
    for tp, dev in RUNS:
        savefile = get_savefile(tp, dev)

        result = benchmark(
            [tp],
            CONFIGS,
            TRACK_INTERVALS,
            NUM_SEEDS,
            [dev],
            steps=STEPS,
            track_events=2,  # does not influence steps, only makes internal check pass
            savefile=savefile,
            header=HEADER,
        )
        print(result)
