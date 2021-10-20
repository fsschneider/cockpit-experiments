"""Quick debugging if histogram 2d performance depends on input data."""

import pprint
import time

import numpy
import torch
from backboard import quantities
from backboard.benchmark.benchmark import run_benchmark
from backboard.quantities.utils_hists import histogram2d, histogram2d_opt
from backboard.utils import fix_deepobs_data_dir
from deepobs.pytorch.config import set_default_device

device = "cpu"
device = "cuda"

params_3c3d = 895210
batch_size = 128

bins = (60, 40)
range = ((-1.0, 1.0), (-1.0, 1.0))


def experiment2():
    """Own histogram2d versus own optimized histogram2d (look at memory consumption)."""
    widths = [0.99, 0.9, 0.5, 0.1, 0.01]

    for w in widths:
        param = (
            torch.from_numpy(numpy.random.uniform(low=-w, high=w, size=(params_3c3d,)))
            .float()
            .to(device)
        )
        batch_grad = (
            torch.from_numpy(
                numpy.random.uniform(low=-w, high=w, size=(batch_size, params_3c3d))
            )
            .float()
            .to(device)
        )

        print(f"Width: {w}")

        for func in [histogram2d_opt]:
            start = time.time()
            func(batch_grad, param, bins, range)

            torch.cuda.synchronize()
            end = time.time()
            print(f"{func.__name__}: {end - start:.5f} s")

        param = param.unsqueeze(0).expand(batch_size, -1).flatten()
        batch_grad = batch_grad.flatten()
        sample = torch.stack((batch_grad, param))

        for func in [
            histogram2d,
            # histogramdd,
        ]:
            start = time.time()
            func(sample, bins, range)

            torch.cuda.synchronize()
            end = time.time()
            print(f"{func.__name__}: {end - start:.5f} s")


def benchmark_histogram_2d():
    testproblem = "cifar10_3c3d"
    track_interval = 1

    def track_schedule(global_step):
        return global_step % track_interval == 0 and global_step >= 0

    steps = 5
    random_seed = 1

    configs = {
        "baseline": [quantities.Time(track_interval=track_interval)],
    }

    for which in [
        "numpy",
        "histogram2d",
        "histogram2d_opt",
        # "histogramdd",
    ]:
        for save_mem in [
            True,
            False,
        ]:
            key = f"BatchGradHistogram2d[save_memory={save_mem}, which={which}]"
            configs[key] = [
                quantities.Time(track_interval=track_interval),
                quantities.BatchGradHistogram2d(
                    track_schedule=track_schedule,
                    save_memory=save_mem,
                    which=which,
                    verbose=True,
                ),
            ]

    runtimes = {}

    for name, quants in configs.items():
        print(name)
        runtimes[name] = run_benchmark(testproblem, quants, steps, random_seed)

    return runtimes


if __name__ == "__main__":

    fix_deepobs_data_dir()

    FORCE_CPU = False
    if FORCE_CPU:
        set_default_device("cpu")

    runtimes = benchmark_histogram_2d()
    print(pprint.pformat(runtimes))

    experiment2()
