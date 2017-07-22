#!/bin/bash
export PYTHONIOENCODING=utf-8
echo "Running mt experiment"
source activate chainer2
python nmt_run.py -o $PWD/out -e $1 -k fisher_train
echo "Finished training mt model"

