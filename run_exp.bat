#!/bin/bash
export PYTHONIOENCODING=utf-8
echo "Running mt experiment"
source activate chainer2
python nmt_run.py -o $PWD/mfcc_out -e $1 -k fisher_train -y 1 -m $2
echo "Finished training mt model"

# longjob -28day -c ./"run_exp.bat 10 1"
# python nmt_run.py -o $PWD/mfcc_out -e 1 -k fisher_train -y 1 -m 1