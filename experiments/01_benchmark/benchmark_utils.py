"""Utility Functions for the Benchmark Plotting."""

import matplotlib as mpl
import pandas as pd
import seaborn as sns
from deepobs.config import DATA_SET_NAMING, TP_NAMING

# add entries for dummy ImageNet network
DATA_SET_NAMING["dummyimagenet"] = "ImageNet (synthetic)"
TP_NAMING["resnet50nobn"] = "ResNet 50 (no batch norm)"


def read_data(filepath):
    """Read the benchmarking data to a pandas DataFrame.

    Args:
        filepath (str): Path to the .csv file holding the data.

    Returns:
        pandas.DataFrame: DataFrame holding the individual runs of the benchmark.
    """
    # CSV file starts with soft- & hardware info that is filtered out
    df = pd.read_csv(filepath, comment="#", index_col=[0])

    # Split by testproblem
    # create unique list of names
    testproblem_set = df.testproblem.unique()
    # create a data frame dictionary to store your data frames
    df_dict = {elem: pd.DataFrame for elem in testproblem_set}
    for key in df_dict.keys():
        df_dict[key] = df[:][df.testproblem == key]

    return df_dict, testproblem_set


def _set_plotting_params():
    # Settings:
    plot_size_default = [16, 8]
    plot_scale = 1.0
    sns.set_style("darkgrid")
    sns.set_context("talk", font_scale=1.2)
    # Apply the settings
    mpl.rcParams["figure.figsize"] = [plot_scale * e for e in plot_size_default]


def _fix_tp_naming(tp):
    dataset = tp.split("_", 1)[0]
    problem = tp.split("_", 1)[1]

    return DATA_SET_NAMING[dataset] + " " + TP_NAMING[problem]


def _fix_dev_naming(dev):
    mapping = {
        "cuda": "GPU",
        "cpu": "CPU",
    }
    return mapping[dev]


def _quantity_naming(quantity, textsc=False):
    quantity_naming = {
        "TICDiag": "TIC\nDiag",
        "HessTrace": "Hess\nTrace",
        "UpdateSize": "Update\nSize",
        "baseline": "Base-\nline",
        # "Loss": "\nLoss",
        "GradNorm": "Grad\nNorm",
        "NormTest": "Norm\nTest",
        "OrthoTest": "Ortho\nTest",
        "InnerTest": "Inner\nTest",
        "GradHist1d": "Grad\nHist1d",
        "GradHist2d": "Grad\nHist2d",
        "Distance": "Dis-\ntance",
        # "InnerProductTest": "Inner\nProduct Test",
        # "OrthogonalityTest": "Orthogonality\nTest",
        # "NormTest": "Norm Test",
        # "baseline": "Baseline",
        # "TICDiag": "TIC",
        # "AlphaOptimized": "Alpha",
        # "BatchGradHistogram1d": "1D\nHistogram",
        # "BatchGradHistogram2d": "2D\nHistogram",
        # "GradNorm": "Gradient\nNorm",
    }

    name = quantity_naming[quantity] if quantity in quantity_naming else quantity

    if textsc:
        split_name = name.splitlines()
        for idx, line in enumerate(split_name):
            split_name[idx] = r"\textsc{" + line + r"}"
        name = "\n".join(split_name)

    return name
