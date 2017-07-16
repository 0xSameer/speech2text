import os
import sys
import argparse
import json
import pickle

program_descrp = """
create a training data dict
"""

'''
example:
export JOSHUA=/afs/inf.ed.ac.uk/group/project/lowres/work/installs/fisher-callhome-corpus
python prep_map_kaldi_segments.py -m $JOSHUA -o $PWD/out/kaldi_segment_map.dict

'''

def main():
    parser = argparse.ArgumentParser(description=program_descrp)
    parser.add_argument('-m','--map_loc', help='fisher-callhome-corpus path',
                        required=True)
    parser.add_argument('-o','--out_path', help='output path',
                        required=True)
    args = vars(parser.parse_args())
    map_loc = args['map_loc']
    out_path = args['out_path']

    if not os.path.exists(map_loc):
        print("fisher-callhome-corpus path given={0:s} does not exist".format(
                                                        map_loc))
        return 0

    # create output file directory:
    if not os.path.exists(out_name):
        print("{0:s} does not exist. Exiting".format(out_path))
        return 0

    kaldi_segment_map_path = os.path.join(out_name,'kaldi_segment_map.dict')

    if not os.path.exists(kaldi_segment_map_path):
        print("{0:s} does not exist. Exiting".format(kaldi_segment_map_path))
        return 0

    print("loading kaldi_segment_map from={0:s}".format(kaldi_segment_map_path))
    kaldi_segment_map = pickle.load(open(kaldi_segment_map_path, "rb"))

    en_map_dir = os.path.join(map_loc,"mapping")
    out_dict = {}

    for cat in kaldi_segment_map:
        print(cat)

    print("Reading json files from: {0:s}".format(kaldi_out_dir))

    for cat in kaldi_segment_map:
        print(cat)

    print("all done ...")


if __name__ == "__main__":
    main()