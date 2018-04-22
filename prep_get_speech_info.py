# coding: utf-8

import os
import sys
import argparse
import json
import pickle
import re
from tqdm import tqdm
import numpy as np

program_descrp = """
get speech durations, transcript and translation lengths
"""

'''
example:
export SPEECH_FEATS=/afs/inf.ed.ac.uk/group/project/lowres/work/corpora/fisher_kaldi/fbank
python prep_get_speech_info.py -o $PWD/out/

'''
def get_info(cat_dict, cat_speech_path, split=False):
    print("reading speech data from: {0:s}".format(cat_speech_path))

    uttr_info = {}

    cnt = 0
    for utt_id in tqdm(cat_dict):
        if not split:
            utt_sp_path = os.path.join(cat_speech_path,
                                       "{0:s}.npy".format(utt_id))
        else:
            utt_sp_path = os.path.join(cat_speech_path, utt_id.split('_',1)[0],
                                       "{0:s}.npy".format(utt_id))

        if os.path.isfile(utt_sp_path):
            uttr_info[utt_id] = {}
            # load speech data
            sp_data = np.load(utt_sp_path)
            # get speech frames
            uttr_info[utt_id]['sp'] = sp_data.shape[0]
            # get text details
            uttr_info[utt_id]['es_w'] = len(cat_dict[utt_id]['es_w'])
            uttr_info[utt_id]['es_c'] = len(cat_dict[utt_id]['es_c'])
            if tuple == type(cat_dict[utt_id]['en_w']):
                # for dev and test, there are multiple translations
                # get max size
                uttr_info[utt_id]['en_w'] = max([len(i) for i in cat_dict[utt_id]['en_w']])
                uttr_info[utt_id]['en_c'] = max([len(i) for i in cat_dict[utt_id]['en_c']])
            else:
                uttr_info[utt_id]['en_w'] = len(cat_dict[utt_id]['en_w'])
                uttr_info[utt_id]['en_c'] = len(cat_dict[utt_id]['en_c'])
        # end if utterance file exists

    # end for utt_id in map dict
    return uttr_info

    print("done...")



def main():
    parser = argparse.ArgumentParser(description=program_descrp)
    parser.add_argument('-o','--out_path', help='output path',
                        required=True)

    args = vars(parser.parse_args())
    out_path = args['out_path']

    # create output file directory:
    if not os.path.exists(out_path):
        print("{0:s} does not exist. Exiting".format(out_path))
        return 0

    # load map dictionary
    map_dict_path = os.path.join(out_path,'new_map.dict')

    if not os.path.exists(map_dict_path):
        print("{0:s} does not exist. Exiting".format(map_dict_path))
        return 0

    print("-"*50)
    print("loading map_dict from={0:s}".format(map_dict_path))
    map_dict = pickle.load(open(map_dict_path, "rb"))
    print("-"*50)

    uttr_info_dict = {}

    for cat in map_dict:
        if "fisher" not in cat:
            continue
        cat_speech_path = os.path.join(out_path, cat)
        if not os.path.isdir(cat_speech_path):
            print("{0:s} does not exist. Exiting!".format(cat_speech_path))
            return 0
        split = "train" in cat
        uttr_info_dict[cat] = get_info(map_dict[cat], cat_speech_path, split)
    # end for category: dev, dev2, test, train

    uttr_info_dict_path = os.path.join(out_path,'info.dict')
    print("-"*50)
    print("saving info dict in: {0:s}".format(uttr_info_dict_path))
    pickle.dump(uttr_info_dict, open(uttr_info_dict_path, "wb"))
    print("all done ...")

if __name__ == "__main__":
    main()
