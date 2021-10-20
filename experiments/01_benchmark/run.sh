#!/usr/bin/env bash

# Run benchmark experiments
for run in {1..20}; do python run_individual.py; done
for run in {1..20}; do python run_grid.py; done
