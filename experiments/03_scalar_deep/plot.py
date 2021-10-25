"""Generate subplots showing the regularization effect of noise."""

import glob
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from cockpit import CockpitPlotter

sys.path.append(os.getcwd())
from experiments.utils.plotting import TikzExport  # noqa

COLORS = sns.color_palette("tab10")
HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)
OUTDIR = os.path.join(HEREDIR, "output")


def read_data(path):
    """Read the data and return the necessary quantities.

    Args:
        path (str): Path to the json logfile

    Returns:
        Data frame with logged metrics.
    """
    plotter = CockpitPlotter()
    plotter._read_tracking_results(path)

    return plotter.tracking_data


def create_loss_plot(data, labels):
    """Create plot of loss over iteration.

    Args:
        data ([pd.DataFrame]): List of DataFrames with data.
        labels ([str]): List of strings with optimizer names.
    """
    plt.figure()
    plt.semilogx()

    for idx, (d, label) in enumerate(zip(data, labels)):
        d_clean = d[["iteration", "Loss"]].dropna()
        plt.plot(
            d_clean["iteration"],
            d_clean["Loss"],
            linewidth=3,
            color=COLORS[idx],
            label=label,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Mini-Batch Loss")
    plt.legend()
    plt.tight_layout()

    out_file = os.path.join(OUTDIR, "Loss")
    TikzExport().save_fig(out_file, png_preview=True, tex_preview=False)
    plt.close()


def create_hess_max_ev_plot(data, labels):
    """Create plot of hess max ev over iteration.

    Args:
        data ([pd.DataFrame]): List of DataFrames with data.
        labels ([str]): List of strings with optimizer names.
    """
    plt.figure()
    plt.semilogx()

    for idx, (d, label) in enumerate(zip(data, labels)):
        d_clean = d[["iteration", "HessMaxEV"]].dropna()
        plt.plot(
            d_clean["iteration"],
            d_clean["HessMaxEV"],
            linewidth=3,
            color=COLORS[idx],
            label=label,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Maximum Hessian eigenvalue")
    plt.legend()
    plt.tight_layout()

    out_file = os.path.join(OUTDIR, "HessMaxEV")
    TikzExport().save_fig(out_file, png_preview=True, tex_preview=False)
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    # find files
    logfile_pattern = "random_seed__42__*__log.json"

    # SGD
    path_sgd = os.path.join(
        HEREDIR,
        "results",
        "scalar_deep",
        "SGD",
        "num_epochs__100000__batch_size__95__l2_reg__0.e+00__lr__1.e-01"
        + "__momentum__0.e+00__nesterov__False",
    )
    file_sgd = glob.glob(os.path.join(path_sgd, logfile_pattern))
    assert len(file_sgd) == 1, f"Found multiple candidates for SGD {file_sgd}"
    file_sgd = file_sgd[0].replace(".json", "")

    # GD
    path_gd = path_sgd.replace("batch_size__95", "batch_size__100")
    file_gd = glob.glob(os.path.join(path_gd, logfile_pattern))
    assert len(file_gd) == 1, f"Found multiple candidates for GD {file_gd}"
    file_gd = file_gd[0].replace(".json", "")

    # Abuse CockpitPlotter to read data
    sgd_data = read_data(file_sgd)
    gd_data = read_data(file_gd)

    # plot
    data = [sgd_data, gd_data]
    labels = ["SGD", "GD"]

    plt.rcParams["axes.facecolor"] = "white"
    create_loss_plot(data, labels)
    create_hess_max_ev_plot(data, labels)
