#!/bin/bash
export PYTHONIOENCODING=utf-8
echo "Running mt experiment"
source activate chainer2
python nmt_run.py -o $PWD/out -e $1 -k fisher_train -y 1
echo "Finished training mt model"

# longjob -28day -c ./"run_exp.bat 20"
# python nmt_run.py -o $PWD/out -e 1 -k fisher_train -y 1