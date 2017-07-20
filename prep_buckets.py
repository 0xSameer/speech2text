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
create buckets for data based on length/duration
"""

'''
example:
export SPEECH_FEATS=/afs/inf.ed.ac.uk/group/project/lowres/work/corpora/fisher_kaldi/fbank
python prep_buckets.py -o $PWD/out/ -n 20 -w 200 -k sp

'''
def display_buckets(buckets_info, cat):
    print("showing buckets for category: {0:s}".format(cat))
    width_b = buckets_info[cat]['width_b']
    num_b = buckets_info[cat]['num_b']
    print('number of buckets={0:d}, width of each bucket={1:d}'.format(width_b, num_b))
    print("{0:5s} | {1:5s} | {2:6s}".format("index", "width", "num"))

    for i, b in enumerate(buckets_info[cat]['buckets']):
        print("{0:5d} | {1:5d} | {2:6d}".format(i, i*width_b, len(b)))
    # end for
# end def

def create_buckets(cat_dict, num_b, width_b, key):
    print("creating buckets for key: {0:s}".format(key))

    # dict to store buckets information
    # create empty buckets and store params
    buckets_info = {'buckets': [[] for i in range(num_b)],
                    'num_b': num_b,
                    'width_b': width_b}

    # loop over all utterances and categorize them into respective buckets
    for utt_id in tqdm(cat_dict):
        bucket = min(cat_dict[utt_id][key] // width_b, num_b-1)
        buckets_info['buckets'][bucket].append(utt_id)
    # end for

    return buckets_info
    print("done...")
# end def



def main():
    parser = argparse.ArgumentParser(description=program_descrp)
    parser.add_argument('-o','--out_path', help='output path',
                        required=True)

    parser.add_argument('-n','--num_b', help='number of buckets',
                        required=True)

    parser.add_argument('-w','--width_b', help='width of buckets',
                        required=True)

    parser.add_argument('-k','--key', help='category to use for length/duration',
                        required=True)

    args = vars(parser.parse_args())
    out_path = args['out_path']
    num_b = int(args['num_b'])
    width_b = int(args['width_b'])
    key = args['key']

    # create output file directory:
    if not os.path.exists(out_path):
        print("{0:s} does not exist. Exiting".format(out_path))
        return 0

    # load map dictionary
    info_dict_path = os.path.join(out_path,'info.dict')

    if not os.path.exists(info_dict_path):
        print("{0:s} does not exist. Exiting".format(info_dict_path))
        return 0

    print("-"*50)
    print("loading info_dict from={0:s}".format(info_dict_path))
    info_dict = pickle.load(open(info_dict_path, "rb"))
    print("-"*50)

    bucket_dict = {}

    for cat in info_dict:
        print("creating buckets for: {0:s}".format(cat))
        bucket_dict[cat] = create_buckets(info_dict[cat], num_b, width_b, key)
    # end for category: dev, dev2, test, train

    # save buckets info
    bucket_dict_path = os.path.join(out_path,'buckets_{0:s}.dict'.format(key))
    print("-"*50)
    print("saving info dict in: {0:s}".format(bucket_dict_path))
    pickle.dump(bucket_dict, open(bucket_dict_path, "wb"))
    print("all done ...")

    # display bucket stats
    for cat in bucket_dict:
        print("-"*50)
        display_buckets(bucket_dict, cat)

if __name__ == "__main__":
    main()

'''
# for each key, we create a separate bucket list
keys = ['en_c', 'en_w', 'es_c', 'es_w', 'sp']



'''