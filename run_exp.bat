#!/bin/bash
export PYTHONIOENCODING=utf-8
echo "Running mt experiment"
source activate chainer3
python nmt_run.py -m $1 -e $2
echo "Finished training mt model"

# python nmt_run.py -m sp_cfg_1 -e 1
# longjob -28day -c ./"run_exp.bat $PWD/sp_cfg_1 10"
# ./run_exp.bat $PWD/sp_cfg_1 10
