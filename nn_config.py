from basics import *
#---------------------------------------------------------------------
# Data Parameters
#---------------------------------------------------------------------
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
#------------------------------------------------------------------------------

print("translating es to en")

# model_dir = "noise_scale_weightdecay"
model_dir = "singlefilter"
EXP_NAME_PREFIX = ""

print("callhome es-en word level configuration")

input_dir = "../../corpora/callhome/uttr_fa_vad_wavs"

INPUT_FEAT_MFCC1_FBANK0 = False

if INPUT_FEAT_MFCC1_FBANK0:
    print("Using MFCCs")
    speech_dir = os.path.join(input_dir, "kaldi", "mfcc_cmvn_dd_vad")
    SPEECH_DIM = 39
else:
    print("Using Filterbanks")
    speech_dir = os.path.join(input_dir, "kaldi", "fbank_vad")
    SPEECH_DIM = 36

MAX_SPEECH_LEN = 400
MIN_SPEECH_LEN = 16
text_data_dict = os.path.join(input_dir, "text_split.dict")
same_spkr_text_data_dict = os.path.join(input_dir, "same_speaker_text_split.dict")

speech_extn = "_fa_vad.std.mfcc"

MODEL_RNN = 0
MODEL_CNN = 1

MODEL_TYPE = MODEL_CNN

lstm1_or_gru0 = False

CHAR_LEVEL = False

OPTIMIZER_ADAM1_SGD_0 = False

CROSS_SPEAKER = False

USE_DROPOUT=False

DROPOUT_RATIO=0.2

ADD_NOISE=True

NOISE_STDEV=0.2

WEIGHT_DECAY=True

if WEIGHT_DECAY:
    WD_RATIO=0.005
else:
    WD_RATIO=0

# ------------------------------------------
NUM_EPOCHS = 100
gpuid = 3
# ------------------------------------------

NUM_SENTENCES = 17394
# use 90% of the data for training

NUM_TRAINING_SENTENCES = 13137
NUM_MINI_TRAINING_SENTENCES = 13137

ITERS_TO_SAVE = 20

NUM_DEV_SENTENCES = 2476
NUM_MINI_DEV_SENTENCES = 2476

NUM_TEST_SENTENCES = 1781

if NUM_MINI_TRAINING_SENTENCES < NUM_TRAINING_SENTENCES:
    SHUFFLE_BATCHES = False
else:
    SHUFFLE_BATCHES = True


if MODEL_TYPE == MODEL_RNN:
    num_layers_enc = 4
    num_layers_dec = 1
elif MODEL_TYPE == MODEL_CNN:
    num_layers_enc = 1
    num_layers_dec = 2

use_attn = SOFT_ATTN
hidden_units = 512
embedding_units = 512

# cnn filter specs - tuple: (kernel size, pad, num filters)
# for now keeping kernel widths as odd
# this keeps the output size the same as the input
cnn_num_channels = 50
cnn_filter_gap = 10
cnn_filter_start = 9
cnn_filter_end = 29
cnn_k_widths = [i for i in range(cnn_filter_start,
                                 cnn_filter_end+1,
                                 cnn_filter_gap)]

cnn_filters = [{"ndim": 1,
                "in_channels": SPEECH_DIM,
                "out_channels": cnn_num_channels,
                "ksize": k,
                "stride": 1,
                "pad": k //2} for k in cnn_k_widths]

num_highway_layers = 2
max_pool_stride = 50
max_pool_pad = 0

print("cnn details:")
for d in cnn_filters:
    print(d)

#------------------------------------------------------------------------------

# if MODEL_TYPE == MODEL_RNN:
#     EXP_NAME_PREFIX += "rnn"
# elif MODEL_TYPE == MODEL_CNN:
#     EXP_NAME_PREFIX += "cnn"
# else:
#     EXP_NAME_PREFIX += "UNK"


if CHAR_LEVEL:
    EXP_NAME_PREFIX += "_char"
else:
    EXP_NAME_PREFIX += "_word"

if lstm1_or_gru0:
    EXP_NAME_PREFIX += "_lstm"
else:
    EXP_NAME_PREFIX += "_gru"

# if OPTIMIZER_ADAM1_SGD_0:
#     EXP_NAME_PREFIX += "_adam"
# else:
#     EXP_NAME_PREFIX += "_adam"

if CROSS_SPEAKER:
    EXP_NAME_PREFIX += "_xspkr"
else:
    EXP_NAME_PREFIX += "_sspkr"

if USE_DROPOUT:
    EXP_NAME_PREFIX += "_drpt-{0:.1f}".format(DROPOUT_RATIO)
else:
    EXP_NAME_PREFIX += "_drpt-0"

if ADD_NOISE:
    EXP_NAME_PREFIX += "_noise-{0:.2f}".format(NOISE_STDEV)
else:
    EXP_NAME_PREFIX += "_noise-0"

if WEIGHT_DECAY:
    EXP_NAME_PREFIX += "_l2-{0:.3f}".format(WD_RATIO)
else:
    EXP_NAME_PREFIX += "_l2-0"

EXP_NAME_PREFIX += "_cnn-num{0:d}-range{1:d}-{2:d}-{3:d}-pool{4:d}".format(
                                                    cnn_num_channels,
                                                    cnn_filter_start,
                                                    cnn_filter_end,
                                                    cnn_filter_gap,
                                                    max_pool_stride*10)

# A total of 11 buckets, with a length range of 7 each, giving total
# BUCKET_WIDTH * NUM_BUCKETS = 77 for e.g.
BUCKET_WIDTH = 3 if not CHAR_LEVEL else 3
NUM_BUCKETS = 14 if not CHAR_LEVEL else 30
TEXT_BUCKETS = [[] for i in range(NUM_BUCKETS)]

MAX_EN_LEN = 150 if not CHAR_LEVEL else 300
#------------------------------------------------
# WARNING !!!!!!!!!!!!!!!!!!!!!!!!
#------------------------------------------------
# SPEECH_BUCKET_WIDTH should be a multiple of 8
#------------------------------------------------
SPEECH_BUCKET_WIDTH = 16
#------------------------------------------------
SPEECH_NUM_BUCKETS = 75

BATCH_SIZE_LOOKUP = {'train':{}, 'dev':{}, 'test':{}}

for i in range(SPEECH_NUM_BUCKETS):
    if i < 50:
        BATCH_SIZE_LOOKUP['train'][i] = 64
    else:
        BATCH_SIZE_LOOKUP['train'][i] = 32

BATCH_SIZE_LOOKUP['dev'] = {}
DEV_SPEECH_BUCKET_WIDTH = 12
DEV_SPEECH_NUM_BUCKETS = 60

for i in range(DEV_SPEECH_NUM_BUCKETS):
    if i < 50:
        BATCH_SIZE_LOOKUP['dev'][i] = 12
    else:
        BATCH_SIZE_LOOKUP['dev'][i] = 8


# create separate widths for input and output, speech and english words/chars
MAX_PREDICT_LEN = BUCKET_WIDTH*NUM_BUCKETS

vocab_path = os.path.join(input_dir, "vocab.dict" if not CHAR_LEVEL else "char_vocab.dict")
w2i_path = os.path.join(input_dir, "w2i.dict" if not CHAR_LEVEL else "char_w2i.dict")
i2w_path = os.path.join(input_dir, "i2w.dict" if not CHAR_LEVEL else "char_i2w.dict")

text_fname = {"en": os.path.join(input_dir, "train.en"), "fr": os.path.join(input_dir, "speech_train.es")}

dev_fname = {"en": os.path.join(input_dir, "dev.en"), "fr": os.path.join(input_dir, "speech_dev.es")}

test_fname = {"en": os.path.join(input_dir, "test.en"), "fr": os.path.join(input_dir, "speech_test.es")}

speech_bucket_data_fname = os.path.join(model_dir, "speech_buckets.dict")

bucket_data_fname = os.path.join(model_dir, "buckets_{0:d}.list" if not CHAR_LEVEL else "buckets_{0:d}_char.list")


if os.path.exists(w2i_path):
    w2i = pickle.load(open(w2i_path, "rb"))
    i2w = pickle.load(open(i2w_path, "rb"))
    vocab = pickle.load(open(vocab_path, "rb"))
    vocab_size_en = min(len(i2w["en"]), max_vocab_size["en"])
    vocab_size_fr = min(len(i2w["fr"]), max_vocab_size["fr"])
    print("vocab size, en={0:d}, fr={1:d}".format(vocab_size_en, vocab_size_fr))

load_existing_model = True

xp = cuda.cupy if gpuid >= 0 else np

name_to_log = "sen-{0:d}_hwy{1:d}-dec{2:d}_emb-{3:d}-h-{4:d}_{5:s}".format(
                                    NUM_MINI_TRAINING_SENTENCES,
                                    num_highway_layers,
                                    num_layers_dec,
                                    embedding_units,
                                    hidden_units,
                                    EXP_NAME_PREFIX)

log_train_fil_name = os.path.join(model_dir, "train_{0:s}.log".format(name_to_log))
log_dev_fil_name = os.path.join(model_dir, "dev_{0:s}.log".format(name_to_log))
model_fil = os.path.join(model_dir, "seq2seq_{0:s}.model".format(name_to_log))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(input_dir):
    print("Input folder not found".format(input_dir))

