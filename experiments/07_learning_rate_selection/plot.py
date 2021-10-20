"""Benchmark Bar Plot of the Overhead of Individual Instruments."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from deepobs.config import DATA_SET_NAMING, TP_NAMING

from cockpit import CockpitPlotter

sys.path.append(os.getcwd())
from experiments.utils.plotting import _get_plot_size, _set_plotting_params  # noqa

# save information
HERE = os.path.abspath(__file__)
DATADIR = os.path.join(os.path.dirname(HERE), "results")
SAVEDIR = os.path.join(os.path.dirname(HERE), "output")
os.makedirs(SAVEDIR, exist_ok=True)


def read_data(dirs):
    """Read the log files of the grid search results.

    Args:
        dirs ([str]): List of test problem folders containing the log files.

    Returns:
        [pandas.DataFrame]: DataFrame holding the performance and alpha statistics.
    """
    # Data Frame for results
    df = pd.DataFrame(
        columns=[
            "testproblem",
            "mean_alpha",
            "median_alpha",
            "std_alpha",
            "learning_rate",
            "train_accuracy",
            "valid_accuracy",
            "test_accuracy",
        ]
    )

    # loop over testproblems
    for d in dirs:
        tp = d.split("/")[-1]
        print("Reading: ", tp)
        # loop over all runs for this problem
        for path, _, files in os.walk(d):
            for name in files:
                if name.endswith("__log.json"):
                    print("Reading: ", name)
                    path = os.path.join(path, name.split(".")[0])
                    # Abuse CockpitPlotter to read data
                    cp = CockpitPlotter()
                    cp._read_tracking_results(path)
                    # Add to dataframe
                    assert (
                        cp.tracking_data.learning_rate.min()
                        == cp.tracking_data.learning_rate.max()
                    )
                    df = df.append(
                        {
                            "testproblem": tp,
                            "mean_alpha": cp.tracking_data.Alpha.mean(),
                            "median_alpha": cp.tracking_data.Alpha.median(),
                            "std_alpha": cp.tracking_data.Alpha.std(),
                            "learning_rate": cp.tracking_data.learning_rate.mean(),
                            "train_accuracy": cp.tracking_data.train_accuracy.iloc[-1],
                            "valid_accuracy": cp.tracking_data.valid_accuracy.iloc[-1],
                            "test_accuracy": cp.tracking_data.test_accuracy.iloc[-1],
                        },
                        ignore_index=True,
                    )

    df = df.sort_values(by=["testproblem", "learning_rate"], ascending=[False, True])
    print(df)
    return df


def plot_data(df):
    """Plot the alpha vs. performance plot.

    Args:
        df (pandas.DataFrame): DataFrame holding the performance and alpha statistics.
    """
    _set_plotting_params()

    fig, ax1 = plt.subplots(
        1,
        1,
        figsize=_get_plot_size(textwidth="neurips", height_ratio=0.22),
    )
    df["log_learning_rate"] = df.learning_rate.map(lambda x: np.log10(x))
    sns.scatterplot(
        ax=ax1,
        data=df,
        x="median_alpha",
        y="test_accuracy",
        hue="testproblem",
        palette="husl",
        size="log_learning_rate",
        zorder=10,
    )

    _beautfiy_plot(ax1)
    # center plot
    # xlims = list(ax1.get_xlim())
    xlims = [-0.65, 0.65]
    xlims[1] = abs(max(xlims, key=abs))
    xlims[0] = -xlims[1]
    ax1.set_xlim(xlims)
    ax1.set_xlabel(r"Median $\alpha$")

    # Zero line
    ax1.axvline(x=0, color="black", lw=0.5)

    # Best run lines
    best_runs = df[
        df.groupby(["testproblem"])["test_accuracy"].transform(max)
        == df["test_accuracy"]
    ]["median_alpha"]
    for idx, best_run in enumerate(best_runs):
        ax1.axvline(x=best_run, lw=0.5, color=sns.husl_palette(len(best_runs))[idx])


def _beautfiy_plot(ax):
    """Beautfiy a given axis of a plot.

    Args:
        ax (mpl.Axes): Axis to beautify.
    """
    # Set labels:
    ax.set_xlabel(ax.xaxis.get_label()._text.title().replace("_", " "))
    ax.set_ylabel(ax.yaxis.get_label()._text.title().replace("_", " "))
    # Set legends
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    new_handles = []
    for handle, label in zip(handles, labels):
        # Manual fix to remove the log learning rate sizes
        if "$" not in label and "learning" not in label:
            try:
                dataset = label.split("_", 1)[0]
                problem = label.split("_", 1)[1]
                if dataset in DATA_SET_NAMING:
                    new_labels.append(
                        DATA_SET_NAMING[dataset] + " " + TP_NAMING[problem]
                    )
                    new_handles.append(handle)
                else:
                    new_labels.append(label.title().replace("_", " "))
                    new_handles.append(handle)
            except IndexError:
                pass
    # Manual ordering to put problems sorted from "easy to hard"
    order = [1, 0, 2, 3]
    new_labels = [new_labels[i] for i in order]
    new_handles = [new_handles[i] for i in order]
    ax.legend(
        new_handles,
        new_labels,
        loc="center right",
        # bbox_to_anchor=(1.22, 0.5),
        ncol=1,
        # framealpha=0.0,
        handletextpad=0.1,
        labelspacing=0.4,
    )


if __name__ == "__main__":
    testproblems = [
        #     "quadratic_deep",
        "mnist_mlp",
        "fmnist_mlp",
        "svhn_3c3d",
        "cifar10_3c3d",
        # "cifar10_3c3dsig"
    ]
    dirs = [os.path.join(DATADIR, tp) for tp in testproblems]

    df = read_data(dirs)

    plot_data(df)

    # Save plot
    plt.savefig(
        os.path.join(SAVEDIR, "median_alpha_vs_performance.pdf"), bbox_inches="tight"
    )
    # plt.show()
