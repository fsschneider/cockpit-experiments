"""Benchmark Bar Plot of the Overhead of Individual Instruments."""

import copy
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import run_individual
import seaborn as sns
from benchmark_utils import _fix_dev_naming, _fix_tp_naming, _quantity_naming, read_data

sys.path.append(os.getcwd())
from experiments.utils.plotting import _get_plot_size, _set_plotting_params  # noqa

HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)
SAVEDIR = os.path.join(HEREDIR, "output/fig_individual/")
os.makedirs(SAVEDIR, exist_ok=True)

PLOT_FRACTION = 0.6
PLOT_HEIGHT = 0.4
APP_PLOT_FRACTION = 0.65
APP_PLOT_FRACTION_EXP = 0.34
APP_PLOT_HEIGHT = 0.13


def plot_data(df, show=True, save=True, title=False, appendix=False):
    """Create a bar plot from the benchmarking data.

    The bar plot shows the relative run time compared to an empty cockpit for
    individual instruments. The run time is averaged over multiple seeds.

    Args:
        df (pandas.DataFrame): DataFrame holding the benchmark data.
        show (bool, optional): Whether to show the plot. Defaults to True.
        save (bool, optional): Whether to save the plot. Defaults to True.
        title (bool, optional): Whether to show a title. Defaults to False.
        appendix (bool, optional): Whether the plot will be used in the appendix.
            Defaults to False.
    """
    fraction = APP_PLOT_FRACTION if appendix else PLOT_FRACTION
    fig, ax = plt.subplots(
        figsize=_get_plot_size(
            textwidth="neurips", fraction=fraction, height_ratio=PLOT_HEIGHT
        )
    )

    # Verify that the data is from a single test problem and use it as a title
    testproblem_set = df.testproblem.unique()
    assert len(testproblem_set) == 1
    tp_name = str(testproblem_set[0])
    tp_name_fixed = _fix_tp_naming(tp_name)

    device_set = df.device.unique()
    assert len(device_set) == 1
    dev_name = str(device_set[0])
    dev_name_fixed = _fix_dev_naming(dev_name)

    ax = plot_data_ax(ax, df)

    if title:
        ax.set_title(
            f"Computational Overhead for {tp_name_fixed} ({dev_name_fixed})",
            fontweight="bold",
        )

    if save:
        savename = "benchmark_" + tp_name + "_" + dev_name
        savename += "_app" if appendix else ""
        savename += ".pdf"
        savepath = os.path.join(SAVEDIR, savename)
        plt.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()


def plot_data_ax(ax, df):
    """Plot the barplot into a given axis.

    Args:
        ax (plt.ax): Axis to plot into.
        df (pandas.DataFrame): DataFrame holding the benchmark data.

    Returns:
        [plt.ax]: Axis to plot into.
    """
    # Smaller font size for quantities
    plt.rcParams.update({"xtick.labelsize": 6})
    width_capsize = 0.25
    width_errorbars = 0.75
    ci = "sd"
    hline_color = "gray"
    hline_style = ":"
    color_palette = "husl"  # "rocket_r", "tab10" "Set2"

    drop = [
        # Remove cockpit_configurations
        "full",
        "business",
        "economy",
        "HessMaxEV",
        "GradHist2d",
    ]

    for d in drop:
        df.drop(df[(df.quantities == d)].index, inplace=True)

    # Compute mean time for basline
    mean_baseline = df.loc[df["quantities"] == "baseline"].mean(axis=0).time_per_step
    df["relative_overhead"] = df["time_per_step"].div(mean_baseline)

    # Order from smallest to largest
    grp_order = df.groupby("quantities").time_per_step.agg("mean").sort_values().index
    # but put "baseline" always in front:
    idx_baseline = np.where(grp_order._index_data == "baseline")[0][0]
    order_list = list(grp_order._index_data)
    order_list.insert(0, order_list.pop(idx_baseline))
    grp_order._index_data = order_list
    grp_order._data = order_list

    sns.barplot(
        x="quantities",
        y="relative_overhead",
        data=df,
        order=grp_order,
        ax=ax,
        capsize=width_capsize,
        errwidth=width_errorbars,
        ci=ci,
        estimator=np.mean,
        palette=color_palette,
    )

    # Line at 1
    ax.axhline(
        y=1,
        color=hline_color,
        linestyle=hline_style,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Run Time Overhead")
    ax.set_xticklabels(_quantity_naming(x.get_text()) for x in ax.get_xticklabels())
    plt.tight_layout()

    # Fix to make the bar plot for the paper a bit more appealing
    ylims = list(ax.get_ylim())
    ylims[1] = max(3.0, ylims[1])
    ax.set_ylim(ylims)

    return ax


def plot_expensive_data(df, show=True, save=True, title=False, appendix=False):
    """Create a bar plot from the expensive instruments.

    The bar plot shows the relative run time compared to an empty cockpit for
    individual instruments. The run time is averaged over multiple seeds.

    Args:
        df (pandas.DataFrame): DataFrame holding the benchmark data.
        show (bool, optional): Whether to show the plot. Defaults to True.
        save (bool, optional): Whether to save the plot. Defaults to True.
        title (bool, optional): Whether to show a title. Defaults to False.
        appendix (bool, optional): Whether the plot will be used in the appendix.
            Defaults to False.
    """
    fig, ax = plt.subplots(
        figsize=_get_plot_size(
            textwidth="neurips",
            fraction=APP_PLOT_FRACTION_EXP,
            height_ratio=PLOT_HEIGHT * APP_PLOT_FRACTION / APP_PLOT_FRACTION_EXP,
        )
    )

    # Verify that the data is from a single test problem and use it as a title
    testproblem_set = df.testproblem.unique()
    assert len(testproblem_set) == 1
    tp_name = str(testproblem_set[0])
    tp_name_fixed = _fix_tp_naming(tp_name)

    device_set = df.device.unique()
    assert len(device_set) == 1
    dev_name = str(device_set[0])
    dev_name_fixed = _fix_dev_naming(dev_name)

    ax = plot_expensive_data_ax(ax, df)

    if title:
        ax.set_title(
            f"Computational Overhead for {tp_name_fixed} ({dev_name_fixed})",
            fontweight="bold",
        )

    if save:
        savename = "benchmark_expensive_" + tp_name + "_" + dev_name
        savename += "_app" if appendix else ""
        savename += ".pdf"
        savepath = os.path.join(SAVEDIR, savename)
        plt.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()


def plot_expensive_data_ax(ax, df):
    """Plot the barplot for the expensive quantities into a given axis.

    Args:
        ax (plt.ax): Axis to plot into.
        df (pandas.DataFrame): DataFrame holding the benchmark data.

    Returns:
        [plt.ax]: Axis to plot into.
    """
    # Plotting Params #
    _set_plotting_params()
    # Smaller font size for quantities
    plt.rcParams.update({"xtick.labelsize": 6})
    width_capsize = 0.25
    width_errorbars = 0.75
    ci = "sd"
    hline_color = "gray"
    hline_style = ":"
    color_palette = "husl"  # "rocket_r", "tab10" "Set2"

    keep = [
        "baseline",
        "HessMaxEV",
        "GradHist2d",
    ]
    drop = [c for c in set(df.quantities) if c not in keep]

    for d in drop:
        df.drop(df[(df.quantities == d)].index, inplace=True)

    # Compute mean time for basline
    mean_baseline = df.loc[df["quantities"] == "baseline"].mean(axis=0).time_per_step
    df["relative_overhead"] = df["time_per_step"].div(mean_baseline)

    # Order from smallest to largest
    grp_order = df.groupby("quantities").time_per_step.agg("mean").sort_values().index
    # but put "baseline" always in front:
    idx_baseline = np.where(grp_order._index_data == "baseline")[0][0]
    order_list = list(grp_order._index_data)
    order_list.insert(0, order_list.pop(idx_baseline))
    grp_order._index_data = order_list
    grp_order._data = order_list

    sns.barplot(
        x="quantities",
        y="relative_overhead",
        data=df,
        order=grp_order,
        ax=ax,
        capsize=width_capsize,
        errwidth=width_errorbars,
        ci=ci,
        estimator=np.mean,
        palette=color_palette,
    )

    # Line at 1
    ax.axhline(
        y=1,
        color=hline_color,
        linestyle=hline_style,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(_quantity_naming(x.get_text()) for x in ax.get_xticklabels())
    plt.tight_layout()

    # Fix to make the bar plot for the paper a bit more appealing
    ylims = list(ax.get_ylim())
    ylims[1] = max(3.0, ylims[1])
    ax.set_ylim(ylims)

    return ax


def plot_combined_app(df, show=True, save=True):
    """Plot both the regular and the expensive data into a single figure.

    Args:
        df (pandas.DataFrame): DataFrame holding the benchmark data.
        show (bool, optional): Whether to show the plot. Defaults to True.
        save (bool, optional): Whether to save the plot. Defaults to True.
    """
    # Plotting Params #
    _set_plotting_params()

    fig, ax = plt.subplots(
        figsize=_get_plot_size(
            textwidth="neurips",
            fraction=1.0,
            height_ratio=APP_PLOT_HEIGHT,
            subplots=(2, 1),
        ),
        ncols=2,
        gridspec_kw={"width_ratios": [APP_PLOT_FRACTION, APP_PLOT_FRACTION_EXP]},
    )
    fig.tight_layout()

    # Verify that the data is from a single test problem and use it as a title
    testproblem_set = df.testproblem.unique()
    assert len(testproblem_set) == 1
    tp_name = str(testproblem_set[0])

    device_set = df.device.unique()
    assert len(device_set) == 1
    dev_name = str(device_set[0])

    ax[0] = plot_data_ax(ax[0], copy.deepcopy(df))

    ax[1] = plot_expensive_data_ax(ax[1], copy.deepcopy(df))

    if save:
        savename = "benchmark_combined_" + tp_name + "_" + dev_name
        savename += ".pdf"
        savepath = os.path.join(SAVEDIR, savename)
        plt.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()


if __name__ == "__main__":
    # Plotting Params #
    _set_plotting_params()

    PLOT_APPENDIX = True

    # # Main Plot
    # MAIN_PROBLEM = ("cifar10_3c3d", "cuda")
    # MAIN_PROBLEM_FILE = run_individual.get_savefile(*MAIN_PROBLEM)

    # df, testproblem_set = read_data(MAIN_PROBLEM_FILE)
    # plot_data(copy.deepcopy(df[MAIN_PROBLEM[0]]), show=True, save=False)

    # Appendix Plots
    if PLOT_APPENDIX:
        APPENDIX_RUNS = [
            # GPU
            ("mnist_logreg", "cuda"),
            ("mnist_mlp", "cuda"),
            ("cifar10_3c3d", "cuda"),
            ("fmnist_2c2d", "cuda"),
            # CPU
            ("mnist_logreg", "cpu"),
            ("mnist_mlp", "cpu"),
            ("cifar10_3c3d", "cpu"),
            ("fmnist_2c2d", "cpu"),
        ]

        for (testproblem, device) in APPENDIX_RUNS:

            filepath = run_individual.get_savefile(testproblem, device)
            df, testproblem_set = read_data(filepath)

            for tp in testproblem_set:
                plot_combined_app(copy.deepcopy(df[tp]), show=False)
