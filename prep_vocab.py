# coding: utf-8

import os
import sys
import argparse
import json
import pickle
import re

program_descrp = """
create vocabulary dict
"""

'''
example:
python prep_map_sp_es_en.py -o $PWD/out/
'''
def get_vocab_units(train_map_dict, key):
    out = {"w2i":{}, "i2w":{}, "freq":{}}
    # loop over each speaker data
    for spk_id in train_map_dict:
        # loop over each utterance with transcriptions+translations
        for units in train_map_dict[spk_id]:
            for unit in units[key]:
                if unit not in out["w2i"]:
                    out["w2i"][unit] = len(out["w2i"])
                    out["freq"][unit] = 1
                else:
                    out["freq"][unit] += 1
            # end for over words/chars in segment
        # end for line in speech segment
    # end for speaker file
    out["i2w"] = {val:key for key, val in out["w2i"].items()}

    return out

def create_vocab(train_map_dict):
    train_vocab_dict = {"es_w":{}, "es_c":{}, "en_w":{}, "en_c":{}}
    for key in train_vocab_dict:
        print("generating vocab for: {0:s}".format(key))
        train_vocab_dict[key] = get_vocab_units(train_map_dict, key)

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
    for unit in train_map_dict[spk_id]:
        print(unit)
        print(unit["en_w"])
        break
    break


train_vocab_dict = pickle.load(open("out/train_vocab.dict", "rb"))
max(train_vocab_dict["en_w"]["i2w"].keys())

In [43]: train_vocab_dict["en_w"]["w2i"][b"hello"]
Out[43]: 0

In [41]: train_vocab_dict["en_w"]["i2w"][0]
Out[41]: b'hello'
--------------------------------------------------
'''