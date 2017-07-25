from basics import *
import prep_buckets
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

out_path = "./out/"

model_dir = "fisher_text"
EXP_NAME_PREFIX = ""

print("callhome es-en configuration")

# encoder key
# 'es_w', 'es_c', or 'sp', and: # 'en_w', 'en_c', or 'sp'
enc_key = 'es_c'
dec_key = 'en_c'

# ------------------------------------------
NUM_EPOCHS = 110
gpuid = 3
# ------------------------------------------

OPTIMIZER_ADAM1_SGD_0 = False

lstm1_or_gru0 = False

USE_DROPOUT=False

DROPOUT_RATIO=0.2

ADD_NOISE=False

NOISE_STDEV=0.2

WEIGHT_DECAY=True

if WEIGHT_DECAY:
    WD_RATIO=0.001
else:
    WD_RATIO=0


ITERS_TO_SAVE = 10

SHUFFLE_BATCHES = True

num_layers_enc = 1
num_layers_dec = 2

use_attn = SOFT_ATTN
hidden_units = 512
embedding_units = 512
# FBANK speech dimensions
SPEECH_DIM = 69

# cnn filter specs - tuple: (kernel size, pad, num filters)
# for now keeping kernel widths as odd
# this keeps the output size the same as the input
if enc_key == 'sp':
    cnn_num_channels = 50
    cnn_filter_gap = 10
    cnn_filter_start = 9
    cnn_filter_end = 99
    num_highway_layers = 2
    max_pool_stride = 40
    max_pool_pad = 0
    BATCH_SIZE = 16
elif enc_key == 'es_c':
    cnn_num_channels = 300
    cnn_filter_gap = 2
    cnn_filter_start = 1
    cnn_filter_end = 9
    num_highway_layers = 4
    max_pool_stride = 5
    max_pool_pad = 0
    BATCH_SIZE = 64
elif enc_key == 'es_w':
    cnn_num_channels = 100
    cnn_filter_gap = 2
    cnn_filter_start = 1
    cnn_filter_end = 9
    num_highway_layers = 4
    max_pool_stride = 1
    max_pool_pad = 0
    BATCH_SIZE = 64

cnn_k_widths = [i for i in range(cnn_filter_start,
                                 cnn_filter_end+1,
                                 cnn_filter_gap)]


if enc_key == 'sp':
    CNN_IN_DIM = SPEECH_DIM
    num_b = 120
    width_b = 16
elif enc_key == 'es_c':
    CNN_IN_DIM = embedding_units
    num_b = 50
    width_b = 5
else:
    CNN_IN_DIM = embedding_units
    num_b = 20
    width_b = 3

if dec_key == 'en_w':
    MAX_EN_LEN = 80
else:
    MAX_EN_LEN = 250

prep_buckets.buckets_main(out_path, num_b, width_b, enc_key)


cnn_filters = [{"ndim": 1,
                "in_channels": CNN_IN_DIM,
                "out_channels": cnn_num_channels,
                "ksize": k,
                "stride": 1,
                "pad": k //2} for k in cnn_k_widths]

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


if not os.path.exists(out_path):
    print("Input folder not found".format(out_path))

print("-"*50)
# load dictionaries
map_dict_path = os.path.join(out_path,'map.dict')
print("loading dict: {0:s}".format(map_dict_path))
map_dict = pickle.load(open(map_dict_path, "rb"))

vocab_dict_path = os.path.join(out_path, 'train_vocab.dict')
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

print('model file name: {0:s}'.format(model_fil))
print('log file name: {0:s}'.format(log_train_fil_name))
