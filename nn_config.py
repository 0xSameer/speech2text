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

# model_dir = "fisher_sp"
model_dir = "fsh_lr_reg"

EXP_NAME_PREFIX = ""

print("callhome es-en configuration")

# encoder key
# 'es_w', 'es_c', or 'sp', and: # 'en_w', 'en_c', or 'sp'
enc_key = 'sp'
dec_key = 'en_w'

# ------------------------------------------
NUM_EPOCHS = 110
gpuid = 2
# ------------------------------------------

OPTIMIZER_ADAM1_SGD_0 = False

lstm1_or_gru0 = False

USE_DROPOUT=False

DROPOUT_RATIO=0.2

ADD_NOISE=True

NOISE_STDEV=0.2

WEIGHT_DECAY=True

if WEIGHT_DECAY:
    WD_RATIO=0.001
else:
    WD_RATIO=0

LEARNING_RATE = 0.05

ONLY_LSTM = False

ITERS_TO_SAVE = 10

SHUFFLE_BATCHES = True

use_attn = SOFT_ATTN
hidden_units = 512
embedding_units = 512
# FBANK speech dimensions
SPEECH_DIM = 69

# cnn filter specs - tuple: (kernel size, pad, num filters)
# for now keeping kernel widths as odd
# this keeps the output size the same as the input
if enc_key == 'sp':
    cnn_num_channels = 200
    cnn_filter_gap = 10
    cnn_filter_start = 9
    cnn_filter_end = 49
    num_highway_layers = 4
    max_pool_stride = 50
    max_pool_pad = 0
    BATCH_SIZE = 12
elif enc_key == 'es_c':
    cnn_num_channels = 100
    cnn_filter_gap = 2
    cnn_filter_start = 1
    cnn_filter_end = 19
    num_highway_layers = 4
    max_pool_stride = 5
    max_pool_pad = 0
    BATCH_SIZE = 64
elif enc_key == 'es_w':
    cnn_num_channels = 100
    cnn_filter_gap = 2
    cnn_filter_start = 1
    cnn_filter_end = 19
    num_highway_layers = 2
    max_pool_stride = 1
    max_pool_pad = 0
    BATCH_SIZE = 64

# if using CNNs, we can have more parameters as sequences are shorter
# due to max pooling
if ONLY_LSTM == False:
    cnn_k_widths = [i for i in range(cnn_filter_start,
                                 cnn_filter_end+1,
                                 cnn_filter_gap)]
    if enc_key == 'sp':
        num_layers_enc = 1
        num_layers_dec = 2
        CNN_IN_DIM = SPEECH_DIM
        num_b = 64
        width_b = 16
        # num_layers_enc = 2

    elif enc_key == 'es_c':
        num_layers_enc = 1
        num_layers_dec = 2
        CNN_IN_DIM = embedding_units
        num_b = 50
        width_b = 6
    else:
        num_layers_enc = 1
        num_layers_dec = 2
        CNN_IN_DIM = embedding_units
        num_b = 20
        width_b = 3

    if dec_key.endswith('_w'):
        MAX_EN_LEN = 50
    else:
        MAX_EN_LEN = 150

else:
    cnn_k_widths = []
    num_layers_enc = 2
    num_layers_dec = 2

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
        MAX_EN_LEN = 60
    else:
        MAX_EN_LEN = 200


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

CNN_PREFIX = "_cnn-num{0:d}-range{1:d}-{2:d}-{3:d}-pool{4:d}".format(
                                                    cnn_num_channels,
                                                    cnn_filter_start,
                                                    cnn_filter_end,
                                                    cnn_filter_gap,
                                                    max_pool_stride*10)

EXP_NAME_PREFIX += "_LSTM" if ONLY_LSTM else CNN_PREFIX

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
