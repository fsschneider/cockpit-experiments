"""Plot the histograms for different pre-processing strategies."""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import run_histograms
import seaborn as sns

from cockpit import CockpitPlotter
from cockpit.instruments.histogram_2d_gauge import _get_xmargin_histogram_data

sys.path.append(os.getcwd())
from experiments.utils.plotting import TikzExport  # noqa

mpl.use("Agg")

HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)
SAVEDIR = os.path.join(HEREDIR, "output/fig_histogram")
os.makedirs(SAVEDIR, exist_ok=True)


def get_cockpit_plotter(filepath, global_step=0):
    """Use a cockpit plotter to read in the tracked data."""
    cp = CockpitPlotter()
    cp._read_tracking_results(filepath)
    # drop all data except first step
    clean_tracking_data = cp.tracking_data.loc[
        cp.tracking_data["iteration"] == global_step
    ]
    cp.tracking_data = clean_tracking_data

    return cp


def get_out_file(tproblem, suffix=".tex"):
    """Get savefile path."""
    suffix = "" if suffix is None else suffix
    filename = f"{tproblem}{suffix}"

    return os.path.join(SAVEDIR, filename)


def plot_net_paper(problem, color, global_step=0):
    """Create TikZ plots for the paper."""
    filepath = run_histograms.get_out_file(problem)
    filepath = os.path.splitext(filepath)[0]
    cp = get_cockpit_plotter(filepath, global_step=global_step)

    vals, mid_points, bin_size = _get_xmargin_histogram_data(cp.tracking_data)
    start_points = [x - bin_size / 2 for x in mid_points]

    plot_histogram(start_points, vals, bin_size, color)

    TikzExport().save_fig(
        get_out_file(problem, suffix=None), png_preview=True, tex_preview=False
    )
    plt.close()


def plot_histogram(start_points, vals, bin_size, color):
    """Plot the histogram."""
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_facecolor("white")
    ax.barh(
        start_points,
        vals,
        height=bin_size,
        color=color,
        linewidth=0.1,
        log=True,
        left=0.9,
        align="edge",
    )
    ax.set_ylim([min(start_points), max(start_points) + bin_size])

    return fig, ax


if __name__ == "__main__":
    COLORS = sns.color_palette("tab10")[:2]

    for problem, color in zip(run_histograms.PROBLEMS, COLORS):
        plot_net_paper(problem, color)
