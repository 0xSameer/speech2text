# coding: utf-8

import os
import sys
import argparse
import json
import pickle
import re
from stemming.porter2 import stem
from tqdm import tqdm

program_descrp = """
create vocabulary dict
"""
PAD = b"_PAD"
GO = b"_GO"
EOS = b"_EOS"
UNK = b"_UNK"

'''
example:
python prep_vocab.py -o $PWD/out/
'''
def get_vocab_units(train_map_dict, key, stemmify=False):
    out = {"w2i":{}, "i2w":{}, "freq":{}}
    START_VOCAB = [PAD, GO, EOS, UNK]
    for w in START_VOCAB:
        out['w2i'][w] = len(out["w2i"])
        out["freq"][w] = 1

    # loop over each speaker data
    for spk_id in tqdm(train_map_dict, ncols=100):
        # loop over each utterance with transcriptions+translations
        if stemmify == False:
            for unit in train_map_dict[spk_id][key]:
                if unit not in out["w2i"]:
                    out["w2i"][unit] = len(out["w2i"])
                    out["freq"][unit] = 1
                else:
                    out["freq"][unit] += 1
            # end for over words/chars in segment
        else:
            for unit in train_map_dict[spk_id][key]:
                stemmed_unit = (stem(unit.decode())).encode()
                if stemmed_unit not in out["w2i"]:
                    out["w2i"][stemmed_unit] = len(out["w2i"])
                    out["freq"][stemmed_unit] = 1
                else:
                    out["freq"][stemmed_unit] += 1
            # end for over stemming
    # end for line in speech segment
    out["i2w"] = {val:key for key, val in out["w2i"].items()}

    return out

def create_vocab(train_map_dict, stemmify=False):
    train_vocab_dict = {"es_w":{}, "es_c":{}, "en_w":{}, "en_c":{}}
    for key in train_vocab_dict:
        print("generating vocab for: {0:s}".format(key))
        train_vocab_dict[key] = get_vocab_units(train_map_dict, 
                                                key, 
                                                stemmify=stemmify if key.endswith("_w") else False)

    return train_vocab_dict


def main():
    parser = argparse.ArgumentParser(description=program_descrp)

    parser.add_argument('-o','--out_path', help='output path',
                        required=True)

    args = vars(parser.parse_args())
    out_path = args['out_path']


    # check for out directory
    if not os.path.exists(out_path):
        print("{0:s} does not exist. Exiting".format(out_path))
        return 0

    # load map dictionary
    map_dict_path = os.path.join(out_path,'map.dict')

    if not os.path.exists(map_dict_path):
        print("{0:s} does not exist. Exiting".format(map_dict_path))
        return 0

    print("-"*50)
    print("loading map_dict from={0:s}".format(map_dict_path))
    map_dict = pickle.load(open(map_dict_path, "rb"))
    print("-"*50)

    train_vocab_dict = create_vocab(map_dict["fisher_train"])
    train_vocab_dict_path = os.path.join(out_path,'train_vocab.dict')
    print("-"*50)
    print("saving vocab dict in: {0:s}".format(train_vocab_dict_path))
    pickle.dump(train_vocab_dict, open(train_vocab_dict_path, "wb"))

    train_stemmed_vocab_dict = create_vocab(map_dict["fisher_train"], 
                                            stemmify=True)
    train_stemmed_vocab_dict_path = os.path.join(out_path,'train_stemmed_vocab.dict')
    print("-"*50)
    print("saving stemmed vocab dict in: {0:s}".format(train_stemmed_vocab_dict_path))
    pickle.dump(train_stemmed_vocab_dict, open(train_stemmed_vocab_dict_path, "wb"))
    print("len stemmed dict for en words = {0:d}".format(len(train_stemmed_vocab_dict['en_w']['w2i'])))
    print("all done ...")
    

if __name__ == "__main__":
    main()

'''
--------------------------------------------------
quick test code
--------------------------------------------------

map_dict = pickle.load(open("out/map.dict", "rb"))
train_map_dict = map_dict["fisher_train"]
for spk_id in train_map_dict:
    for unit in train_map_dict[spk_id]["en_w"]:
        print(unit)
    break

train_vocab_dict = pickle.load(open("out/train_vocab.dict", "rb"))
max(train_vocab_dict["en_w"]["i2w"].keys())

In [43]: train_vocab_dict["en_w"]["w2i"][b"hello"]
Out[43]: 0

In [41]: train_vocab_dict["en_w"]["i2w"][0]
Out[41]: b'hello'
--------------------------------------------------
'''