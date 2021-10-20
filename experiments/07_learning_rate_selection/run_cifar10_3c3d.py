"""Run SGD on Cifar10 using different learning rates."""

from gridsearch import run

from cockpit.utils import schedules

lrs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
track_schedule = schedules.linear(10)

run("cifar10_3c3d", lrs, track_schedule)
