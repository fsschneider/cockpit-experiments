"""Run SGD on Fashion MNIST using different learning rates."""

from gridsearch import run

from cockpit.utils import schedules

lrs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
track_schedule = schedules.linear(1)

run("fmnist_mlp", lrs, track_schedule)
