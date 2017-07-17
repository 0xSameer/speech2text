# coding: utf-8

import os
import sys
import argparse
import json
import pickle
import re
program_descrp = """
create a training data dict
"""

'''
example:
export JOSHUA=/afs/inf.ed.ac.uk/group/project/lowres/work/installs/fisher-callhome-corpus
python prep_map_sp_es_en.py -m $JOSHUA -o $PWD/out/

'''

# borrowed from Google
# **Based on TensorFlow code:
# https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/data_utils.py**
_WORD_SPLIT = re.compile(b"([.,!?\"':~;)(])")
_CHAR_SPLIT = re.compile(b"([.,!?\":~;)(])")

'''
test_sentence = b"haha though,don't say.what could've been"

basic_tokenizer(test_sentence)
Out: b'haha though don t say what could ve been'

char_tokenizer(test_sentence)
'''
def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.sub(b"", w) for w in _WORD_SPLIT.split(space_separated_fragment))
    return b" ".join([w.lower() for w in words if w])

# UGLY FUNCTION
# ಠ_ಠ
def char_tokenizer(sentence):
    # preserve apostrophe suffixes as chars
    chars = []
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_CHAR_SPLIT.sub(b"", w) for w in _WORD_SPLIT.split(space_separated_fragment))

    words = [w for w in words if w]

    i = 0
    while i < len(words):
        word = words[i]
        if word == b"'":
            if i < len(words)-1:
                apos_str = "{0:s}{1:s}".format("'",words[i+1].decode().lower())
                chars.append(apos_str.encode())
            i+=1
        else:
            char_in_word = [c.encode() for c in word.decode().lower()]
            chars.extend(char_in_word)
        i+=1
        if i < len(words):
            chars.append(b" ")

    return chars[:-1] if len(chars)>0 and chars[-1] == b" " else chars

def get_words_and_chars(file_name):
    es_words = []
    es_chars = []
    with open(file_name, "rb") as in_f:
        for line in in_f:
            es_words.append(basic_tokenizer(line))
            es_chars.append(char_tokenizer(line))
    return es_words, es_chars


def read_map_file(cat, segment_map, map_loc):
    print("-"*50)
    print("aligning: {0:s}".format(cat))
    print("-"*50)

    es_en_map_dir = os.path.join(map_loc,"mapping")
    es_en_fil_dir = os.path.join(map_loc,"corpus","ldc")

    es_en_map_file = os.path.join(es_en_map_dir, cat)
    es_file = os.path.join(es_en_fil_dir, cat+".es")

    print("reading es lines from: {0:s}".format(es_file))
    es_words, es_chars = get_words_and_chars(es_file)

    if "train" in cat:
        en_file = os.path.join(es_en_fil_dir, cat+".en")
        print("reading en lines from: {0:s}".format(en_file))
        en_words, en_chars = get_words_and_chars(en_file)

    else:
        en_words = []
        en_chars = []
        en_words_per_file = {}
        en_chars_per_file = {}

        for i in range(4):
            en_file = os.path.join(es_en_fil_dir, "{0:s}.en.{1:d}".format(cat,i))
            print("reading en lines from: {0:s}".format(en_file))
            en_words_per_file[i], en_chars_per_file[i] = get_words_and_chars(en_file)

        for i in range(len(en_words_per_file[0])):
            try:
                en_words.append(tuple([en_words_per_file[j][i] for j in range(4)]))
                en_chars.append(tuple([en_chars_per_file[j][i] for j in range(4)]))
            except:
                print("HAAAAAALP!!!")
                print([len(en_words_per_file[f]) for f in en_words_per_file])

    # preparing output
    out_dict = {}
    print("reading map file: {0:s}".format(es_en_map_file))
    with open(es_en_map_file, "r") as map_f:
        for i, line in enumerate(map_f):
            fid, uids = line.strip().split()
            uids = uids.split("_")
            # adding speech segments
            speech_ids = []
            for uid in uids:
                map_key = "{0:s}_{1:s}".format(fid, uid)
                speech_ids.append(segment_map[map_key])
            # determine speaker id
            # "20050908_182943_22_fsp-A-000055-000156"
            speech_fil = speech_ids[0]["seg_name"].rsplit("-",2)[0]

            if speech_fil not in out_dict:
                out_dict[speech_fil] = []

            out_dict[speech_fil].append({"es_w": es_words[i],
                                         "es_c": es_chars[i],
                                         "en_w": en_words[i],
                                         "en_c": en_chars[i],
                                         "seg": speech_ids})
        # end for
    # end with open map
    return out_dict

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
    if not os.path.exists(out_path):
        print("{0:s} does not exist. Exiting".format(out_path))
        return 0

    kaldi_segment_map_path = os.path.join(out_path,'kaldi_segment_map.dict')

    if not os.path.exists(kaldi_segment_map_path):
        print("{0:s} does not exist. Exiting".format(kaldi_segment_map_path))
        return 0

    print("loading kaldi_segment_map from={0:s}".format(kaldi_segment_map_path))
    kaldi_segment_map = pickle.load(open(kaldi_segment_map_path, "rb"))

    # prepare map dictionary
    map_dict = {}
    map_dict_path = os.path.join(out_path,'map.dict')

    for cat in kaldi_segment_map:
        map_dict[cat] = read_map_file(cat, kaldi_segment_map[cat], map_loc)

    print("-"*50)
    print("saving map dict in: {0:s}".format(map_dict_path))
    pickle.dump(map_dict, open(map_dict_path, "wb"))
    print("all done ...")

if __name__ == "__main__":
    main()