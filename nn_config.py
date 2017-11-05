from basics import *
import prep_buckets
#------------------------------------------------------------------------------
# Data Parameters
#------------------------------------------------------------------------------
max_vocab_size = {"en" : 200000, "fr" : 200000}

# Special vocabulary symbols - we always put them at the start.
PAD = b"_PAD"
GO = b"_GO"
EOS = b"_EOS"
UNK = b"_UNK"
START_VOCAB = [PAD, GO, EOS, UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

NO_ATTN = 0
SOFT_ATTN = 1

SINGLE_1D_CNN    = 0
DEEP_1D_CNN      = 1
DEEP_2D_CNN      = 2
#------------------------------------------------------------------------------
# FBANK
# out_path = "./fbank_out/"
# ------------------------------------
out_path = "./both_fbank_out"
wavs_path = os.path.join(out_path, "wavs")
# ------------------------------------
# model_dir = "fsh_fbank_10_tf0.5"
# model_dir = "fsh_fbank"
model_dir = "nov17"
# model_dir = "new_vocab_callhome_fbank"
# ------------------------------------
SPEECH_DIM = 40
# ------------------------------------
# MFCC
# out_path = "./mfcc_out/"
# model_dir = "fsh_again"
# SPEECH_DIM = 39
# ------------------------------------
print("fisher + callhome sp/es - en configuration")
# ------------------------------------
# encoder key
# 'es_w', 'es_c', or 'sp', and: # 'en_w', 'en_c', or 'sp'
enc_key = 'sp'
dec_key = 'en_w'

# ------------------------------------
gpuid = 2
# ------------------------------------
# scaling factor for reducing batch
# size
BATCH_SIZE = 256
BATCH_SIZE_MEDIUM = 200
BATCH_SIZE_SMALL = 100
BATCH_SIZE_SCALE = 1
TRAIN_SIZE_SCALE = 1

STEMMIFY = False
BI_RNN = False

FSH1_CH0 = False

RANDOM_SEED_VALUE="{0:s}_{1:d}".format("fsh" if FSH1_CH0 else "callh",
                                       100 // TRAIN_SIZE_SCALE)

EXP_NAME_PREFIX = "" if RANDOM_SEED_VALUE == "haha" else "_{0:s}_".format(RANDOM_SEED_VALUE)
# ------------------------------------

# ------------------------------------
LEARNING_RATE = 1.0
# ------------------------------------
teacher_forcing_ratio = 0.8
# ------------------------------------
OPTIMIZER_ADAM1_SGD_0 = True
# ------------------------------------

# ------------------------------------
WEIGHT_DECAY=True
if WEIGHT_DECAY:
    WD_RATIO=1e-4
else:
    WD_RATIO=0
# ------------------------------------

# ------------------------------------
ITERS_GRAD_NOISE = 0
# default noise function is
# recommended to be either:
# 0.01, 0.3 or 1.0
GRAD_NOISE_ETA = 0.01
# ------------------------------------

# ------------------------------------
USE_DROPOUT=True
DROPOUT_RATIO=0.4
# ------------------------------------

# ------------------------------------
ITERS_TO_SAVE = 5
# ------------------------------------

# ------------------------------------
lstm1_or_gru0 = False
ONLY_LSTM = False
# ------------------------------------
CNN_TYPE = DEEP_2D_CNN
# ------------------------------------
USE_LN = True
USE_BN = True
FINE_TUNE = False
# ------------------------------------

# ------------------------------------
# only applicable for mini mode
SHUFFLE_BATCHES = False
# ------------------------------------

# ------------------------------------
use_attn = SOFT_ATTN
ATTN_W = True
# ------------------------------------

# ------------------------------------
ADD_NOISE=True
if enc_key != 'sp':
    ADD_NOISE=False

NOISE_STDEV=0.2
# ------------------------------------

# ------------------------------------
ITERS_TO_WEIGHT_NOISE = 75
WEIGHT_NOISE_MU = 0.0
WEIGHT_NOISE_SIGMA = 0.01
# ------------------------------------

# ------------------------------------
hidden_units = 256
embedding_units = 256
# ------------------------------------

# if using CNNs, we can have more parameters as sequences are shorter
# due to max pooling
if ONLY_LSTM == False:
    # cnn_k_widths = [i for i in range(cnn_filter_start,
    #                              cnn_filter_end+1,
    #                              cnn_filter_gap)]
    if enc_key == 'sp':
        # ------------------------------------
        num_layers_enc = 3
        num_layers_dec = 3
        # ------------------------------------
        num_highway_layers = 0
        CNN_IN_DIM = SPEECH_DIM
        num_b = 15
        width_b = 150

    elif enc_key == 'es_c':
        num_layers_enc = 3
        num_layers_dec = 3
        num_highway_layers = 2
        CNN_IN_DIM = embedding_units
        num_b = 10
        width_b = 15
    else:
        num_layers_enc = 2
        num_layers_dec = 2
        num_highway_layers = 2
        CNN_IN_DIM = embedding_units
        num_b = 20
        width_b = 3

    if dec_key.endswith('_w'):
        MAX_EN_LEN = 120
    else:
        MAX_EN_LEN = 200

else:
    cnn_k_widths = []
    num_layers_enc = 4
    num_layers_dec = 2
    num_highway_layers = 2

    if enc_key == 'sp':
        print("Cannot train speech using only LSTM")

    elif enc_key == 'es_c':
        CNN_IN_DIM = embedding_units
        num_b = 50
        width_b = 4
    else:
        CNN_IN_DIM = embedding_units
        num_b = 20
        width_b = 3

    if dec_key.endswith('_w'):
        MAX_EN_LEN = 50
    else:
        MAX_EN_LEN = 150


# prepare buckets
prep_buckets.buckets_main(out_path, num_b, width_b, enc_key)

max_pool_stride = 5
max_pool_pad = 0

if CNN_TYPE == SINGLE_1D_CNN:
    cnn_num_channels = 100
    cnn_filters = [{"ndim": 1,
                    "in_channels": CNN_IN_DIM,
                    "out_channels": cnn_num_channels,
                    "ksize": k,
                    "stride": 1,
                    "pad": k // 2} for k in cnn_k_widths]
elif CNN_TYPE == DEEP_1D_CNN:
    # static CNN configuration
    # cnn_filters = [
    #     {"ndim": 1,
    #     "in_channels": CNN_IN_DIM,
    #     "out_channels": 64,
    #     "ksize": 4,
    #     "stride": 2,
    #     "pad": 4 // 2},
    #     {"ndim": 1,
    #     "in_channels": 64,
    #     "out_channels": 64,
    #     "ksize": 4,
    #     "stride": 3,
    #     "pad": 4 // 2},
    # ]
    pass

else:
    # static CNN configuration
    # googlish
    # ------------------------------------
    cnn_filters = [
        {"in_channels": None,
        "out_channels": 32,
        "ksize": (5,3),
        "stride": (3,2),
        "pad": (5 // 2, 3 // 2)},
        {"in_channels": None,
        "out_channels": 32,
        "ksize": (5,3),
        "stride": (3,2),
        "pad": (5 // 2, 3 // 2)},
    ]
    # ------------------------------------

print("cnn details:")
for d in cnn_filters:
    print(d)

#------------------------------------------------------------------------------

EXP_NAME_PREFIX += "_{0:s}_{1:s}".format(enc_key, dec_key)

if lstm1_or_gru0:
    EXP_NAME_PREFIX += "_lstm"
else:
    EXP_NAME_PREFIX += "_gru"

if USE_DROPOUT:
    EXP_NAME_PREFIX += "_drpt-{0:.1f}".format(DROPOUT_RATIO)
else:
    EXP_NAME_PREFIX += "_drpt-0"

# if ADD_NOISE:
#     EXP_NAME_PREFIX += "_noise-{0:.2f}".format(NOISE_STDEV)
# else:
EXP_NAME_PREFIX += "_noise-0"

if WEIGHT_DECAY:
    EXP_NAME_PREFIX += "_l2-{0:.6f}".format(WD_RATIO)
else:
    EXP_NAME_PREFIX += "_l2-0"


if CNN_TYPE == SINGLE_1D_CNN:
    CNN_PREFIX = "_cnn-num{0:d}-range{1:d}-{2:d}-{3:d}-pool{4:d}".format(
                                                    cnn_num_channels,
                                                    cnn_filter_start,
                                                    cnn_filter_end,
                                                    cnn_filter_gap,
                                                    max_pool_stride*10)

elif CNN_TYPE == DEEP_1D_CNN:
    # str_cnn_sizes = "_".join([str(d['out_channels']) for d in cnn_filters])
    # # if sum(cnn_max_pool) // len(cnn_max_pool) > 1:
    # CNN_PREFIX = "_{0:s}_{1:s}_DCNN".format(str_cnn_sizes,
    #                                         "_".join(map(str,cnn_max_pool)))
    str_cnn_sizes = "_".join([str(d['out_channels']) for d in cnn_filters])
    CNN_PREFIX = "_{0:s}_{1:s}_1DCNN".format(str_cnn_sizes,
                                         "_".join([str(i["stride"]) for i in cnn_filters ]))


else:
    str_cnn_sizes = "_".join([str(d['out_channels']) for d in cnn_filters])
    CNN_PREFIX = "_{0:s}_{1:s}_2DCNN".format(str_cnn_sizes,
                                         "_".join([str(i["stride"][0]) for i in cnn_filters ]))

EXP_NAME_PREFIX += "_LSTM" if ONLY_LSTM else CNN_PREFIX

EXP_NAME_PREFIX += "_BN" if USE_BN else ''

EXP_NAME_PREFIX += "_LN" if USE_LN else ''

EXP_NAME_PREFIX += "_STEMMIFY" if STEMMIFY else ''

EXP_NAME_PREFIX += "_BI_RNN" if BI_RNN else ''


EXP_NAME_PREFIX += "_enc-{0:d}".format(num_layers_enc) if num_layers_enc > 1 else ""

if not os.path.exists(out_path):
    print("Input folder not found".format(out_path))

print("-"*50)
# load dictionaries
map_dict_path = os.path.join(out_path,'map.dict')
print("loading dict: {0:s}".format(map_dict_path))
map_dict = pickle.load(open(map_dict_path, "rb"))


if STEMMIFY == False:
    vocab_dict_path = os.path.join(out_path, 'train_vocab.dict')
else:
    vocab_dict_path = os.path.join(out_path, 'train_stemmed_vocab.dict')

print("loading dict: {0:s}".format(vocab_dict_path))
vocab_dict = pickle.load(open(vocab_dict_path, "rb"))
print("-"*50)

bucket_dict_path = os.path.join(out_path,'buckets_{0:s}.dict'.format(enc_key))
print("loading dict: {0:s}".format(bucket_dict_path))
bucket_dict = pickle.load(open(bucket_dict_path, "rb"))
print("-"*50)

for cat in map_dict:
    print('utterances in {0:s} = {1:d}'.format(cat, len(map_dict[cat])))

if enc_key != 'sp':
    vocab_size_es = len(vocab_dict[enc_key]['w2i'])
else:
    vocab_size_es = 0
vocab_size_en = len(vocab_dict[dec_key]['w2i'])
print('vocab size for {0:s} = {1:d}'.format(enc_key, vocab_size_es))
print('vocab size for {0:s} = {1:d}'.format(dec_key, vocab_size_en))

NUM_MINI_TRAINING_SENTENCES = len(map_dict['fisher_train'])

load_existing_model = True

xp = cuda.cupy if gpuid >= 0 else np

if ONLY_LSTM == False:
    name_to_log = "sen-{0:d}_hwy{1:d}-dec{2:d}_emb-{3:d}-h-{4:d}_{5:s}".format(
                                    NUM_MINI_TRAINING_SENTENCES,
                                    num_highway_layers,
                                    num_layers_dec,
                                    embedding_units,
                                    hidden_units,
                                    EXP_NAME_PREFIX)
else:
    name_to_log = "sen-{0:d}_enc{1:d}-dec{2:d}_emb-{3:d}-h-{4:d}_{5:s}".format(
                                    NUM_MINI_TRAINING_SENTENCES,
                                    num_layers_enc,
                                    num_layers_dec,
                                    embedding_units,
                                    hidden_units,
                                    EXP_NAME_PREFIX)


log_train_fil_name = os.path.join(model_dir, "train_{0:s}.log".format(name_to_log))
log_dev_fil_name = os.path.join(model_dir, "dev_{0:s}.log".format(name_to_log))
model_fil = os.path.join(model_dir, "seq2seq_{0:s}.model".format(name_to_log))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

print('model file name: {0:s}'.format(model_fil))
print('log file name: {0:s}'.format(log_train_fil_name))

'''
FBANK:

python kaldi_io.py test_fbank.ark fisher_fbank/fisher_test
python kaldi_io.py dev_fbank.ark fisher_fbank/fisher_dev
python kaldi_io.py dev2_fbank.ark fisher_fbank/fisher_dev2
python kaldi_io.py train_fbank.ark fisher_fbank/fisher_train

export SPEECH_FEATS=/afs/inf.ed.ac.uk/group/project/lowres/work/corpora/fisher_kaldi/fisher_fbank

export JOSHUA=/afs/inf.ed.ac.uk/group/project/lowres/work/installs/fisher-callhome-corpus

export OUT=$PWD/fbank_out

python prep_map_kaldi_segments.py -m $JOSHUA -o $OUT

python prep_map_sp_es_en.py -m $JOSHUA -o $OUT

python prep_speech_segments.py -m $SPEECH_FEATS -o $OUT

python prep_vocab.py -o $OUT

python prep_get_speech_info.py -o $OUT

python nmt_run.py -o $PWD/fbank_out -e 10 -k fisher_train -y 1 -m 1


MFCC:
python kaldi_io.py test_mfcc.ark fisher_mfcc/fisher_test
python kaldi_io.py dev_mfcc.ark fisher_mfcc/fisher_dev
python kaldi_io.py dev2_mfcc.ark fisher_mfcc/fisher_dev2
python kaldi_io.py train_mfcc.ark fisher_mfcc/fisher_train

export SPEECH_FEATS=/afs/inf.ed.ac.uk/group/project/lowres/work/corpora/fisher_kaldi/mfcc

export JOSHUA=/afs/inf.ed.ac.uk/group/project/lowres/work/installs/fisher-callhome-corpus

export OUT=$PWD/mfcc_out

python prep_map_kaldi_segments.py -m $JOSHUA -o $OUT

python prep_map_sp_es_en.py -m $JOSHUA -o $OUT

python prep_speech_segments.py -m $SPEECH_FEATS -o $OUT

python prep_vocab.py -o $OUT

python prep_get_speech_info.py -o $OUT

python nmt_run.py -o $PWD/mfcc_out -e 10 -k fisher_train -y 1 -m 1


CALLHOME:

FBANK:

python kaldi_io.py callhome_test_fbank.ark callhome_fbank/callhome_test
python kaldi_io.py callhome_dev_fbank.ark callhome_fbank/callhome_dev
python kaldi_io.py callhome_train_fbank.ark callhome_fbank/callhome_train

export SPEECH_FEATS=/afs/inf.ed.ac.uk/group/project/lowres/work/corpora/fisher_kaldi/fisher_fbank

export JOSHUA=/afs/inf.ed.ac.uk/group/project/lowres/work/installs/fisher-callhome-corpus

export OUT=$PWD/callhome_fbank_out

python prep_map_kaldi_segments.py -m $JOSHUA -o $OUT

python prep_map_sp_es_en.py -m $JOSHUA -o $OUT

python prep_speech_segments.py -m $SPEECH_FEATS -o $OUT

python prep_vocab.py -o $OUT

python prep_get_speech_info.py -o $OUT

python nmt_run.py -o $PWD/fbank_out -e 10 -k fisher_train -y 1 -m 1

with tqdm(total=total_utts, dynamic_ncols=True) as pbar

---------------------------------------------
export BLEU_SCRIPT=/afs/inf.ed.ac.uk/group/project/lowres/work/installs/mosesdecoder/scripts/generic/multi-bleu.perl

perl $BLEU_SCRIPT fisher_dev_en.ref0 fisher_dev_en.ref1 fisher_dev_en.ref2 fisher_dev_en.ref3 < fisher_dev_mt-output

perl $BLEU_SCRIPT e2e_ast_decode/refs/fisher_dev/sorted-normalized-fisher_dev.en* < e2e_ast_decode/hyps/fisher_dev/fisher_spa_eng_ast_003_base_r0.txt

perl $BLEU_SCRIPT fisher_dev_en.ref* < fisher_dev_mt-output
BLEU = 22.83, 59.1/31.5/17.2/9.6 (BP=0.970, ratio=0.970, hyp_len=38270, ref_len=39454)

perl $BLEU_SCRIPT google_fisher_dev_ref_* < google_fisher_dev_r0.en
BLEU = 46.41, 76.6/55.2/39.4/27.9 (BP=1.000, ratio=1.002, hyp_len=39703, ref_len=39623)

perl $BLEU_SCRIPT google_fisher_dev_r[1-3]* < fisher_dev_mt-output
BLEU = 20.82, 54.7/28.6/15.4/8.4 (BP=0.980, ratio=0.980, hyp_len=38270, ref_len=39050)


'''
