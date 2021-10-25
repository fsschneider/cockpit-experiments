#!/usr/bin/env bash

# SGD
python run.py --batch_size 95 -N 100000 --lr 0.1

# GD
python run.py --batch_size 100 -N 100000 --lr 0.1
