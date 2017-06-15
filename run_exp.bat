#!/bin/bash
export PYTHONIOENCODING=utf-8
echo "Running mt experiment"
source activate chainer2
python nmt_trials.py
echo "Finished training mt model"

