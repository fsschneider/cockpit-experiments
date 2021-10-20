import json
import os

import matplotlib.pyplot as plt
import numpy
import run
import seaborn as sns

from experiments.utils.plotting import TikzExport

HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)

SAVEDIR = os.path.join(HEREDIR, "output")
os.makedirs(SAVEDIR, exist_ok=True)


def get_out_file(device):
    savename = f"{device}"
    return os.path.join(SAVEDIR, savename)


def plot(device):
    COLORS = sns.color_palette("tab10")

    files = [run.get_out_file(hist_func, device) for hist_func in run.hist_funcs]

    if device == "cpu":
        files.append(run.get_numpy_out_file())

    plt.figure()
    plt.yscale("log")
    plt.xlabel("Histogram balance")
    plt.ylabel("Run time [s]")

    for color, f in zip(COLORS, files):
        # load
        with open(f) as json_file:
            benchmark = json.load(json_file)

        # create plotting data
        widths_float = []
        mean_run_times = []
        # std_run_times = []

        widths = sorted(benchmark.keys())
        for w in widths:
            runs = list(benchmark[w].values())

            widths_float.append(float(w))
            mean_run_times.append(numpy.mean(runs))
            # std_run_times.append(numpy.std(runs))

        # plot
        # NUM_STD = 2
        # won't export correctly
        # plt.errorbar(
        #     widths_float,
        #     mean_run_times,
        #     yerr=[NUM_STD * s for s in std_run_times],
        #     color=color,
        #     ls="dashed",
        # )
        plt.plot(
            widths_float,
            mean_run_times,
            label=get_label(f),
            color=color,
            ls="dashed",
        )

    plt.legend()
    TikzExport().save_fig(
        get_out_file(device),
        post_process=True,
        png_preview=True,
        tex_preview=False,
    )

    plt.close()


def get_label(f):
    if "numpy" in f:
        return r"\textsc{NumPy} (single thread)"
    if "histogramdd" in f:
        return r"\textsc{PyTorch} (third party)"
    if "histogram2d" in f:
        return r"\textsc{PyTorch} (\textsc{Cockpit})"
    raise ValueError(f"No label known for file {f}")


def copy_to_tex_dir(device):
    SRC = get_out_file(device) + ".tex"
    DEST = os.path.join(SAVEDIR, "../../../tex/CVPR2021/fig/11_histogram2d/")
    os.makedirs(DEST, exist_ok=True)

    cmd = f"cp -rv {SRC} {DEST}"
    os.system(cmd)


if __name__ == "__main__":
    for device in run.devices:
        plot(device)

    for device in run.devices:
        copy_to_tex_dir(device)
