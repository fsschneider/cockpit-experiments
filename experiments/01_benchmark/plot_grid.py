"""Benchmark Heatmap Plot of the Overhead of Cockpit Configurations."""

import os
import sys

import matplotlib.pyplot as plt
import run_grid
import seaborn as sns
from benchmark_utils import _fix_dev_naming, _fix_tp_naming, read_data

sys.path.append(os.getcwd())
from experiments.utils.plotting import _get_plot_size, _set_plotting_params  # noqa

HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)
SAVEDIR = os.path.join(HEREDIR, "output/fig_grid/")
os.makedirs(SAVEDIR, exist_ok=True)

PLOT_FRACTION = 0.35
PLOT_FRACTION_OTHER = 0.6
APP_PLOT_FRACTION = 0.49
PLOT_HEIGHT = 0.4
PLOT_HEIGHT_APP = 0.75
PLOT_HEIGHT_IMAGENET = 0.5
PLOT_FRACTION_IMAGENET = 0.4


def plot_data(
    ax,
    df,
    cbar=True,
    hide_y_axis=False,
    ext_title=False,
    show_title=False,
    reduced_config=False,
):
    """Create a heatmap plot from the benchmarking data.

    The heatmap plot shows the relative run time compared to an empty cockpit for
    different cockpit configurations (e.g. 'economy', 'business' and 'full') and
    different tracking rates. The run time is averaged over multiple seeds.

    Args:
        ax (mpl.axes): Axes for the plot.
        df (pandas.DataFrame): DataFrame holding the benchmark data.
        cbar (book, optional): Whether to show a colorbar. Defaults to True.
        hide_y_axis (bool, optional): Whether to hide labels and ticks on the y-axis.
            Defaults to False.
        ext_title (bool, optional): Whether to show the full title including the
            super title. Defaults to False.
        show_title (bool, optional): Whether to show the figure's title. Defaults to
            False.
        reduced_config (bool, optional): Whether only the `baseline` and `economy`
            configs are tested. Currently this is the case for `ImageNet`.
            Defaults to False.
    """
    # Plotting Params #
    cmap = "vlag"  # , "tab10" "Set2"
    cockpit_configs = (
        ["baseline", "economy"]
        if reduced_config
        else ["baseline", "economy", "business", "full"]
    )
    annot = True
    vmin, vmax = 1, 3

    # Verify that the data is from a single test problem and use it as a title
    testproblem_set = df.testproblem.unique()
    assert len(testproblem_set) == 1
    tp_name = _fix_tp_naming(str(testproblem_set[0]))

    device_set = df.device.unique()
    assert len(device_set) == 1
    dev_name = _fix_dev_naming(str(device_set[0]))

    # Only keep cockpit configurations
    df = df.loc[df["quantities"].isin(cockpit_configs)]

    # reshape and average
    df = (
        df.groupby(["quantities", "track_interval"])
        .mean()
        .unstack(level=0)["time_per_step"]
    ).T

    # Sort index by cockpit_configs list
    df = df.reindex(cockpit_configs)

    # relative overhead
    ref_value = df.loc["baseline", 1]
    df = df.divide(ref_value)

    hm = sns.heatmap(
        df,
        cmap=cmap,
        ax=ax,
        annot=annot,
        vmin=vmin,
        vmax=vmax,
        lw=0.2,
        center=2,
        cbar=cbar,
        annot_kws={"fontsize": 5},
    )
    if cbar:
        cbar_ticks = list(range(vmin, vmax + 1))
        hm.collections[0].colorbar.ax.tick_params(size=1, width=0.0)
        hm.collections[0].colorbar.set_ticks(cbar_ticks)
        hm.collections[0].colorbar.set_ticklabels([str(t) for t in cbar_ticks])

    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
    hm.set_xticklabels(hm.get_xticklabels(), rotation=0)

    ax.tick_params(
        axis="both",
        which="major",
        labelbottom=False,
        bottom=False,
        top=False,
        labeltop=True,
        pad=-4,
    )

    if ext_title:
        add_title = "Computational Overhead for"
    else:
        add_title = ""
    title = f"{add_title}\n{tp_name} ({dev_name})"
    if show_title:
        ax.set_title(title, fontsize=8)
    ax.set_xlabel("Track Interval")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Configuration")
    if hide_y_axis:
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()


def plot_individual(
    df,
    tp,
    dev,
    show=True,
    save=True,
    show_title=True,
    appendix=False,
    reduced_config=False,
):
    """Plot the heatmaps of each test problem individually.

    Args:
        df (pandas.DataFrame): DataFrame holding the benchmark data.
        tp (str): Name of the testproblem.
        dev (str): Device.
        show (bool, optional): Switch to show the plot. Defaults to True.
        save (bool, optional): Switch to save the plot as pdf. Defaults to True.
        show_title (bool, optional): Switch to show  title. Defaults to False.
        appendix (bool, optional): Whether the plot will be used in the appendix.
            Defaults to False.
        reduced_config (bool, optional): Whether only the `baseline` and `economy`
            configs are tested. Currently this is the case for `ImageNet`.
            Defaults to False.
    """
    height_ratio = PLOT_HEIGHT * PLOT_FRACTION_OTHER / PLOT_FRACTION
    fraction = PLOT_FRACTION
    if appendix:
        height_ratio = PLOT_HEIGHT_APP

    if reduced_config:
        height_ratio = PLOT_HEIGHT_IMAGENET
        fraction = PLOT_FRACTION_IMAGENET

    fig, ax = plt.subplots(
        figsize=_get_plot_size(
            textwidth="neurips",
            fraction=fraction,
            height_ratio=height_ratio,
        )
    )
    plot_data(
        ax,
        df,
        show_title=show_title,
        cbar=not appendix,
        reduced_config=reduced_config,
    )

    # Save plot
    if save:
        savename = "benchmark_" + tp + "_" + dev
        savename += "_app" if appendix else ""
        savename += ".pdf"
        savepath = os.path.join(SAVEDIR, savename)
        plt.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()


def plot_all(dev, show=True, save=True):
    """Plot the heatmap of three test problems in a single figure.

    Args:
        dev (str): Device to plot.
        show (bool, optional): Switch to show the plot. Defaults to True.
        save (bool, optional): Switch to save the plot as pdf. Defaults to True.
    """
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    tps = [
        "quadratic_deep",
        "mnist_mlp",
        "cifar10_3c3d",
    ]

    for idx, tp in enumerate(tps):
        df, testproblem_set = read_data(run_grid.get_savefile(tp, dev))

        for tp in testproblem_set:
            if idx == 2:
                plot_data(axs[2], df[tp], cbar=True, hide_y_axis=True)
            elif idx == 1:
                plot_data(axs[1], df[tp], cbar=False, hide_y_axis=True, ext_title=True)
            elif idx == 0:
                plot_data(axs[0], df[tp], cbar=False)
    plt.tight_layout()

    # Save plot
    if save:
        savepath = os.path.join(SAVEDIR, "heatmap.pdf")
        plt.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()


if __name__ == "__main__":
    _set_plotting_params()

    PLOT_APPENDIX = True

    # Main Plot
    MAIN_PROBLEM = ("cifar10_3c3d", "cuda")
    MAIN_PROBLEM_FILE = run_grid.get_savefile(*MAIN_PROBLEM)

    df, testproblems = read_data(MAIN_PROBLEM_FILE)
    plot_individual(df[testproblems[0]], *MAIN_PROBLEM, show=False, show_title=False)

    # Appendix Plots
    if PLOT_APPENDIX:
        APPENDIX_RUNS = [
            # GPU
            ("mnist_logreg", "cuda"),
            ("mnist_mlp", "cuda"),
            ("cifar10_3c3d", "cuda"),
            ("fmnist_2c2d", "cuda"),
            ("dummyimagenet_resnet50nobn", "cuda"),
            # CPU
            ("mnist_logreg", "cpu"),
            ("mnist_mlp", "cpu"),
            ("cifar10_3c3d", "cpu"),
            ("fmnist_2c2d", "cpu"),
        ]

        for (testproblem, device) in APPENDIX_RUNS:

            filepath = run_grid.get_savefile(testproblem, device)
            df, _ = read_data(filepath)

            plot_individual(
                df[testproblem],
                testproblem,
                device,
                show=False,
                show_title=False,
                appendix=True,
                reduced_config=True
                if testproblem == "dummyimagenet_resnet50nobn"
                else False,
            )
