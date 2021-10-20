"""Utility functions for plotting the experiments."""

import os
import re
import subprocess

import matplotlib.pyplot as plt
import seaborn as sns
from tikzplotlib import save as tikz_save


def _set_plotting_params():
    """Set some consistent plotting settings and styles."""
    # Seaborn settings:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.0)

    # Matplotlib settings (using tex font)
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
        ],  # Use Times New Roman, and Times as a back up
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 7,
        "font.size": 7,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        # Fix for missing \mathdefault command
        "text.latex.preamble": r"\newcommand{\Mathdefault}[1][]{}",
        # Less space between label and axis
        "xtick.major.pad": 0.0,
        "ytick.major.pad": 0.0,
        # More space between subplots
        "figure.subplot.hspace": 0.3,
        # Less space around the plot
        "savefig.pad_inches": 0.0,
        # dashed grid lines
        "grid.linestyle": "dashed",
        # width of grid lines
        "grid.linewidth": 0.4,
        # Show thin edge around each plot
        "axes.edgecolor": "black",
        "axes.linewidth": 0.4,
    }
    plt.rcParams.update(tex_fonts)


def _get_plot_size(
    textwidth="cvpr", fraction=1.0, height_ratio=(5 ** 0.5 - 1) / 2, subplots=(1, 1)
):
    r"""Returns the matplotlib plot size to fit with the LaTeX textwidth.

    Args:
        textwidth (float or string, optional): LaTeX textwidth in pt. Can be accessed
            directly in LaTeX via `\the\textwidth` and needs to be replaced
            accoring to the used template. Defaults to "cvpr", which automatically uses
            the size of the CVPR template.
        fraction (float, optional): Fraction of the textwidth the plot should occupy.
            Defaults to 1.0 meaning a full width figure.
        height_ratio (float, optional): Ratio of the height to width. A value of
            0.5 would result in a figure that is half as high as it is wide.
            Defaults to the golde ratio.
        subplots (array-like, optional): The number of rows and columns of subplots.
            Defaults to (1,1)

    Returns:
        [float]: Desired height and width of the matplotlib figure.
    """
    if textwidth == "cvpr":
        width = 496.85625
    elif textwidth == "cvpr_col":
        width = 237.13594
    elif textwidth == "neurips":
        width = 397.48499
    else:
        width = textwidth

    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height = fig_width * height_ratio * (subplots[0] / subplots[1])

    return fig_width, fig_height


class TikzExport:
    """Handle matplotlib export to TikZ."""

    def __init__(self, extra_axis_parameters=None):
        """Initialize.

        Note:
        -----
        Extra axis parameters are inserted in alphabetical order.
        By prepending 'z' to the style, it will be inserted last.
        Like that, you can overwrite the previous axis parameters
        in 'zmystyle'.

        Args:
            extra_axis_parameters (list): Extra axis options to be passed
                (as a list or set) to pgfplots. Default is None.
        """
        if extra_axis_parameters is None:
            extra_axis_parameters = {"zmystyle"}

        self.extra_axis_parameters = extra_axis_parameters

    def save_fig(
        self,
        out_file,
        fig=None,
        png_preview=True,
        tex_preview=True,
        override_externals=True,
        post_process=True,
    ):
        """Save matplotlib figure as TikZ. Optional PNG out."""
        if fig is not None:
            self.set_current(fig)

        tex_file = self._add_extension(out_file, "tex")

        out_dir = os.path.dirname(out_file)
        os.makedirs(out_dir, exist_ok=True)

        if png_preview:
            png_file = self._add_extension("{}-preview".format(out_file), "png")
            plt.savefig(png_file, bbox_inches="tight")

        tikz_save(
            tex_file,
            override_externals=override_externals,
            extra_axis_parameters=self.extra_axis_parameters,
        )

        if post_process is True:
            self.post_process(out_file)

        if tex_preview is True:
            tex_preview_file = self._add_extension("{}-preview".format(out_file), "tex")
            with open(tex_file, "r") as f:
                content = "".join(f.readlines())

            preamble = r"""\documentclass[tikz]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{amssymb}
\usetikzlibrary{shapes,
                pgfplots.groupplots,
                shadings,
                calc,
                arrows,
                backgrounds,
                colorbrewer,
                shadows.blur}
% customize "zmystyle" as you wish
 \pgfkeys{/pgfplots/zmystyle/.style={
         % legend pos = north east,
         % xmin=1, xmax=20,
         % ymin = 1, ymax = 1.2,
         % title = {The title},
     }
 }
\begin{document}
"""

            postamble = r"\end{document}"
            preview_content = preamble + content + postamble

            with open(tex_preview_file, "w") as f:
                f.write(preview_content)

            subprocess.run(["pdflatex", "-output-directory", out_dir, tex_preview_file])

    def save_subplots(
        self,
        out_path,
        names,
        fig=None,
        png_preview=True,
        tex_preview=True,
        override_externals=True,
        post_process=True,
    ):
        """Save subplots of figure into single TikZ figures."""
        if fig is None:
            fig = plt.gcf()

        axes = self.axes_as_individual_figs(fig)

        for name, subplot in zip(names, axes):
            assert len(subplot.get_axes()) == 1

            out_file = os.path.join(out_path, name)
            self.save_fig(
                out_file,
                fig=subplot,
                png_preview=png_preview,
                tex_preview=tex_preview,
                override_externals=override_externals,
                post_process=post_process,
            )

    @staticmethod
    def set_current(fig):
        """Set current figure."""
        plt.figure(fig.number)

    def post_process(self, tikz_file):
        """Remove from matplotlib2tikz export what should be configurable."""
        file = self._add_extension(tikz_file, "tex")
        with open(file, "r") as f:
            content = f.readlines()

        content = self._remove_linewidths(content)
        content = self._remove_some_arguments(content)

        joined_content = "".join(content)

        joined_content = self._remove_by_regex(joined_content)

        with open(file, "w") as f:
            f.write(joined_content)

    @staticmethod
    def _add_extension(filename, extension, add_to_filename=None):
        if add_to_filename is None:
            add_to_filename = ""
        return "{}{}.{}".format(filename, add_to_filename, extension)

    @staticmethod
    def _remove_linewidths(lines):
        """Remove line width specifications."""
        linewidths = [
            r"ultra thick",
            r"very thick",
            r"semithick",
            r"thick",
            r"very thin",
            r"ultra thin",
            r"thin",
        ]
        new_lines = []
        for line in lines:
            for width in linewidths:
                line = line.replace(width, "")
            new_lines.append(line)
        return new_lines

    @staticmethod
    def _remove_some_arguments(lines):
        """Remove lines containing certain specifications."""
        # remove lines containing these specifications
        to_remove = [
            r"legend cell align",
            # r"legend style",
            r"x grid style",
            r"y grid style",
            # r"tick align",
            # r"tick pos",
            r"ytick",
            # r"xtick",
            r"yticklabels",
            # r"xticklabels",
            # "ymode",
            r"log basis y",
        ]

        for pattern in to_remove:
            lines = [line for line in lines if pattern not in line]

        return lines

    @staticmethod
    def _remove_by_regex(joined_content):
        """."""
        reg_patterns = [
            r"xticklabels=[\S\s]*?(\},\n)",
            r"xtick=[\S\s]*?(\},\n)",
        ]

        for pat in reg_patterns:
            joined_content = re.sub(pat, "", joined_content)

        return joined_content

    @staticmethod
    def axes_as_individual_figs(fig):
        """Return a list of figures, each containing a single axes.

        `fig` is messed up during this procedure as the axes are being removed
        and inserted into other figures.
        Note: MIGHT BE UNSTABLE
        -----
        https://stackoverflow.com/questions/6309472/
        matplotlib-can-i-create-axessubplot-objects-then-add-them-to-a-figure-instance
        Axes deliberately aren't supposed to be shared between different figures now.
        As a workaround, you could do this fig2._axstack.add(fig2._make_key(ax), ax),
        but it's hackish and likely to change in the future.
        It seems to work properly, but it may break some things.

        Args:
            fig (figure): Matplotlib figure.

        Returns:
            [figures]: List of individual figures.
        """
        fig_axes = fig.get_axes()

        # breaks fig
        for ax in fig_axes:
            fig.delaxes(ax)

        fig_list = []
        for ax in fig_axes:
            new_fig = plt.figure()
            new_fig._axstack.add(new_fig._make_key(ax), ax)
            new_fig.axes[0].change_geometry(1, 1, 1)
            fig_list.append(new_fig)

        return fig_list
