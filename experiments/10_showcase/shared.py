"""Shared code among showcase experiments."""
import glob
import math
import os

from PIL import Image
from torch.optim import SGD

import cockpit.utils.schedules as schedules
from cockpit import CockpitPlotter

HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)


optimizer_class = SGD
hyperparams = {
    "lr": {"type": float, "default": 0.1},
    "momentum": {"type": float, "default": 0.0},
    "nesterov": {"type": bool, "default": False},
}


def cosine_decay_restarts_schedule(
    steps_for_cycle,
    max_epochs,
    increase_restart_interval_factor=2,
):
    """Cyclical cosine decay learning rate schedule."""
    factors = cosine_decay_restarts(
        steps_for_cycle,
        max_epochs,
        increase_restart_interval_factor=increase_restart_interval_factor,
    )
    return lambda epoch: factors[epoch]


def cosine_decay_restarts(
    steps_for_cycle,
    max_epochs,
    increase_restart_interval_factor=2,
):
    """Learning rate factors for cyclic schedule with restarts."""
    lr_factors = []

    step = 0
    cycle = 0

    for _ in range(0, max_epochs + 1):
        step += 1
        completed_fraction = step / steps_for_cycle
        cosine_decayed = 0.5 * (1 + math.cos(math.pi * completed_fraction))
        lr_factors.append(cosine_decayed)

        if completed_fraction == 1:
            step = 0
            cycle += 1
            steps_for_cycle = steps_for_cycle * increase_restart_interval_factor

    return lr_factors


plot_schedule = schedules.logarithmic(0, 19, steps=300, base=2)
track_schedule = plot_schedule


def locate_json_log(testproblem, optimizer_class):
    """Locate json logfile."""
    RUN_DIR = os.path.join(HEREDIR, "results", testproblem, optimizer_class.__name__)
    RUN_PATTERN = os.path.join(RUN_DIR, "*/*__log.json")
    RUN_MATCH = glob.glob(RUN_PATTERN)
    assert len(RUN_MATCH) == 1, f"Found no or multiple files: {RUN_MATCH}"
    return RUN_MATCH[0]


def plot_to_tex_dir(testproblem, optimizer_class):
    """Plot final screen to tex directory.

    Note: This function only works if the file calling it is in the
       same directory as this script.

    Args:
        testproblem (str): DeepOBS Testproblem.
        optimizer_class (class): Optimizer Class.
    """
    source = locate_json_log(testproblem, optimizer_class)
    dest = os.path.join(HEREDIR, f"../../tex/fig/10_showcase/{testproblem}.png")

    print(f"[exp10|plot] Using log: {source}")
    print(f"[exp10|plot] Saving to: {dest}")

    # plot
    plotter = CockpitPlotter()
    plotter.plot(
        source, block=False, show_plot=False, show_log_iter=True, save_plot=False
    )
    plotter.fig.savefig(dest)


def animate(testproblem, optimizer_class):
    """Build an animation from the logged .json file."""
    logpath = locate_json_log(testproblem, optimizer_class)
    savedir = os.path.dirname(logpath)

    plotter = CockpitPlotter()

    # regenerate plots
    plotter._read_tracking_results(logpath)
    track_events = list(plotter.tracking_data["iteration"])

    frame_paths = []
    for idx, global_step in enumerate(track_events):
        print(f"Plotting {idx:05d}/{len(track_events):05d}")

        plotter.plot(
            logpath,
            show_plot=False,
            save_plot=False,
            block=False,
            show_log_iter=True,
            discard=global_step,
        )

        this_frame_path = os.path.join(savedir, f"animation_frame_{idx:05d}.png")
        plotter.fig.savefig(this_frame_path)
        frame_paths.append(this_frame_path)

    frame, *frames = [Image.open(f) for f in frame_paths]

    animation_savepath = os.path.join(savedir, "showcase.gif")

    # Collect images and create Animation
    frame.save(
        fp=animation_savepath,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=200,
        loop=0,
    )
