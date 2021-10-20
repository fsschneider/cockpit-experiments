import glob
import os

from shared import DIR, parse, report_memory, run, set_up, skip_if_exists

from cockpit.quantities import GradHist1d


class BatchGradHistogram1dExpensive(GradHist1d):
    """One-dimensional histogram of individual gradient elements.

    Computes all individual gradients before creating a histogram.
    """

    def extension_hooks(self, global_step):
        """Return list of BackPACK extension hooks required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            [callable]: List of required BackPACK extension hooks for the current
                iteration.
        """
        return self._adapt.extension_hooks(global_step)

    def _compute(self, global_step, params, batch_loss):
        """Evaluate the individual gradient histogram.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            dict: Entry ``'hist'`` holds the histogram, entry ``'edges'`` holds
                the bin limits.
        """
        hist, edges = None, None

        for p in params:
            hist_p, edges_p = self._compute_histogram(p.grad_batch)

            hist = hist_p if hist is None else hist + hist_p
            edges = edges_p if edges is None else edges

            del p.grad_batch

        return {"hist": hist, "edges": edges}


def get_out_files(testproblem):
    """Return all available output files for a test problem."""
    pattern = os.path.join(DIR, f"{testproblem}_expensive_*.csv")
    return glob.glob(pattern)


def out_file(testproblem, num_run):
    """Return save path for a specific run of a test problem."""
    return os.path.join(DIR, f"{testproblem}_expensive_{num_run:03d}.csv")


if __name__ == "__main__":
    set_up()

    testproblem, num_run = parse()
    filename = out_file(testproblem, num_run)

    skip_if_exists(filename)

    def benchmark_fn():
        def track_schedule(global_step):
            return True

        quants = [BatchGradHistogram1dExpensive(track_schedule)]
        run(quants, testproblem)

    data = report_memory(benchmark_fn)
    data.to_csv(filename)
