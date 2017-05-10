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

EXP_NAME_PREFIX="es_en_speech2text"


print("callhome es-en word level configuration")

input_dir = "../../corpora/callhome/uttr_fa_vad_wavs"

speech_dir = os.path.join(input_dir, "mfb_std")
text_data_dict = os.path.join(input_dir, "text_split.dict")

speech_extn = "_fa_vad.std.mfb"

CHAR_LEVEL = True

NUM_SENTENCES = 17394
# use 90% of the data for training
NUM_TRAINING_SENTENCES = 13137
# NUM_TRAINING_SENTENCES = 1000
NUM_MINI_DEV_SENTENCES = 200
ITERS_TO_SAVE = 5
NUM_DEV_SENTENCES = 2476
NUM_TEST_SENTENCES = 1781
BATCH_SIZE = 20
# A total of 11 buckets, with a length range of 7 each, giving total
# BUCKET_WIDTH * NUM_BUCKETS = 77 for e.g.
BUCKET_WIDTH = 3 if not CHAR_LEVEL else 3
NUM_BUCKETS = 14 if not CHAR_LEVEL else 30
MAX_PREDICT_LEN = BUCKET_WIDTH*NUM_BUCKETS

vocab_path = os.path.join(input_dir, "vocab.dict" if not CHAR_LEVEL else "char_vocab.dict")
w2i_path = os.path.join(input_dir, "w2i.dict" if not CHAR_LEVEL else "char_w2i.dict")
i2w_path = os.path.join(input_dir, "i2w.dict" if not CHAR_LEVEL else "char_i2w.dict")

print("translating es to en")
model_dir = "es_en_model_char"

text_fname = {"en": os.path.join(input_dir, "train.en"), "fr": os.path.join(input_dir, "speech_train.es")}

dev_fname = {"en": os.path.join(input_dir, "dev.en"), "fr": os.path.join(input_dir, "speech_dev.es")}

test_fname = {"en": os.path.join(input_dir, "test.en"), "fr": os.path.join(input_dir, "speech_test.es")}

EXP_NAME= "{0:s}_callhome_es_en".format(EXP_NAME_PREFIX)

bucket_data_fname = os.path.join(model_dir, "buckets_{0:d}.list" if not CHAR_LEVEL else "buckets_{0:d}_char.list")

if os.path.exists(w2i_path):
    w2i = pickle.load(open(w2i_path, "rb"))
    i2w = pickle.load(open(i2w_path, "rb"))
    vocab = pickle.load(open(vocab_path, "rb"))
    vocab_size_en = min(len(i2w["en"]), max_vocab_size["en"])
    vocab_size_fr = min(len(i2w["fr"]), max_vocab_size["fr"])
    print("vocab size, en={0:d}, fr={1:d}".format(vocab_size_en, vocab_size_fr))

num_layers_enc = 4
num_layers_dec = 4
use_attn = SOFT_ATTN
hidden_units = 512

gpuid = -1