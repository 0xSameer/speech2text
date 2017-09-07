#!/bin/bash
export PYTHONIOENCODING=utf-8
echo "Running mt experiment"
source activate chainer2
python nmt_run.py -o $1 -e $2 -k fisher_train -y 1 -m $3
echo "Finished training mt model"

# longjob -28day -c ./"run_exp.bat 10 1"
# python nmt_run.py -o $PWD/mfcc_out -e 1 -k fisher_train -y 1 -m 1