#!/bin/bash
export PYTHONIOENCODING=utf-8
echo "Running mt experiment"
source activate chainer3
python nmt_run.py -o $1 -e $2 -k $3 -y 1 -m $4
echo "Finished training mt model"

# longjob -28day -c ./"run_exp.bat $PWD/fbank_out 10 fisher_train 0"
# longjob -28day -c ./"run_exp.bat $PWD/mfcc_out 10 fisher_train 1"
# python nmt_run.py -o $PWD/callhome_fbank_out -e 1 -k fisher_train -y 1 -m 1
# export OUT=$PWD/callhome_fbank_out
# longjob -28day -c ./"run_exp.bat $PWD/callhome_fbank_out 10 fisher_train 0"
# longjob -28day -c ./"run_exp.bat $PWD/callhome_fbank_out 10 callhome_train 0"


# longjob -28day -c ./"run_exp.bat $PWD/both_fbank_out 10 fisher_train 0"
# longjob -28day -c ./"run_exp.bat $PWD/both_fbank_out 10 callhome_train 0"
# ./run_exp.bat $PWD/both_fbank_out 10 fisher_train 0
# ./run_exp.bat $PWD/both_fbank_out 10 callhome_train 0