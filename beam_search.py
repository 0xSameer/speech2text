
# coding: utf-8

# In[1]:

import os
import sys
import argparse
from nmt_run import *


program_descrp = """
beam search
"""

parser = argparse.ArgumentParser(description=program_descrp)

parser.add_argument('-o','--nmt_path', help='model path',
                    required=True)

args = vars(parser.parse_args())
cfg_path = args['nmt_path']

# cfg_path = "interspeech/sp_20hrs"

print("-"*80)
print("Using model: {0:s}".format(cfg_path))
print("-"*80)

def get_utt_data(eg_utt, curr_set):
    # get shape
    local_input_path = os.path.join(m_cfg['data_path'], curr_set)

    width_b = bucket_dict[dev_key]["width_b"]
    num_b = bucket_dict[dev_key]["num_b"]
    utt_list = [eg_utt]

    batch_data = get_batch(map_dict[curr_set],
                           enc_key,
                           dec_key,
                           utt_list,
                           vocab_dict,
                           num_b * width_b,
                           200,
                           input_path=local_input_path)

    return batch_data


# In[8]:


last_epoch, model, optimizer, m_cfg, t_cfg = check_model(cfg_path)

train_key = m_cfg['train_set']
dev_key = m_cfg['dev_set']
batch_size=t_cfg['batch_size']
enc_key=m_cfg['enc_key']
dec_key=m_cfg['dec_key']
input_path = os.path.join(m_cfg['data_path'], m_cfg['dev_set'])
# -------------------------------------------------------------------------
# get data dictionaries
# -------------------------------------------------------------------------
map_dict, vocab_dict, bucket_dict = get_data_dicts(m_cfg)
batch_size = {'max': 1, 'med': 1, 'min': 1, 'scale': 1}

# In[9]:


random.seed("meh")
# random.seed("haha")

# Eval parameters
ref_index = -1
min_len, max_len= 0, m_cfg['max_en_pred']
# min_len, max_len = 0, 10
displayN = 50
m_dict=map_dict[dev_key]
# wavs_path = os.path.join(m_cfg['data_path'], "wavs")
wavs_path = os.path.join("../chainer2/speech2text/both_fbank_out/", "wavs")
v_dict = vocab_dict['en_w']
key = m_cfg['dev_set']



def get_encoder_states():
    rnn_states = {"c": [], "h": []}
    # ---------------------------------------------------------------------
    # get the hidden and cell state (LSTM) of the first RNN in the decoder
    # ---------------------------------------------------------------------
    if model.m_cfg['bi_rnn']:
        for i, (enc, rev_enc) in enumerate(zip(model.rnn_enc,
                                     model.rnn_rev_enc)):
            h_state = F.concat((model[enc].h, model[rev_enc].h))
            rnn_states["h"].append(h_state)
            if model.m_cfg['rnn_unit'] == RNN_LSTM:
                c_state = F.concat((model[enc].c, model[rev_enc].c))
                rnn_states["c"].append(c_state)
    else:
        for enc, dec in zip(model.rnn_enc, model.rnn_dec):
            rnn_states["h"].append(model[enc].h)
            if model.m_cfg['rnn_unit'] == RNN_LSTM:
                rnn_states["c"].append(model[enc].c)
            # end if
        # end for all layers
    # end if bi-rnn
    return rnn_states
    # ---------------------------------------------------------------------


# In[15]:


def get_decoder_states():
    rnn_states = {"c": [], "h": []}
    # ---------------------------------------------------------------------
    # get the hidden and cell state (LSTM) of the first RNN in the decoder
    # ---------------------------------------------------------------------
    for i, dec in enumerate(model.rnn_dec):
        rnn_states["h"].append(model[dec].h)
        if model.m_cfg['rnn_unit'] == RNN_LSTM:
            rnn_states["c"].append(model[dec].c)
        # end if
    # end for all layers
    return rnn_states
    # ---------------------------------------------------------------------


# In[16]:


def set_decoder_states(rnn_states):
    # ---------------------------------------------------------------------
    # set the hidden and cell state (LSTM) for the decoder
    # ---------------------------------------------------------------------
    for i, dec in enumerate(model.rnn_dec):
        if model.m_cfg['rnn_unit'] == RNN_LSTM:
            model[dec].set_state(rnn_states["c"][i], rnn_states["h"][i])
        else:
            model[dec].set_state(rnn_states["h"][i])
        # end if
    # end for all layers
    # ---------------------------------------------------------------------


# In[17]:


def encode_utt_data(X):
    # get shape
    batch_size = X.shape[0]
    # encode input
    model.forward_enc(X)


# In[18]:


def init_hyp():
    beam_entry = {"hyp": [GO_ID], "score": 0}
    beam_entry["dec_state"] = get_encoder_states()
    a_units = m_cfg['attn_units']
    ht = Variable(xp.zeros((1, a_units), dtype=xp.float32))
    beam_entry["attn_v"] = ht
    return beam_entry



def decode_beam_step(decode_entry, beam_width=3):
    xp = cuda.cupy if model.gpuid >= 0 else np

    with chainer.using_config('train', False):

        word_id, dec_state, attn_v = (decode_entry["hyp"][-1],
                                        decode_entry["dec_state"],
                                        decode_entry["attn_v"])

        # set decoder state
        set_decoder_states(dec_state)
        #model.set_decoder_state()

        # intialize starting word symbol
        #print("beam step curr word", v_dict['i2w'][word_id].decode())
        curr_word = Variable(xp.full((1,), word_id, dtype=xp.int32))

        prob_out = {}
        prob_print_str = []

        # -----------------------------------------------------------------
        # decode and predict
        pred_out, ht = model.decode(curr_word, attn_v)
        # -----------------------------------------------------------------
        # printing conditional probabilities
        # -----------------------------------------------------------------
        pred_probs = xp.asnumpy(F.log_softmax(pred_out).data[0])
        top_n_probs = xp.argsort(pred_probs)[-beam_width:]

        new_entries = []

        curr_dec_state = get_decoder_states()

        for pi in top_n_probs[::-1]:
            #print("{0:10s} = {1:5.4f}".format(v_dict['i2w'][pi].decode(), pred_probs[pi]))
            new_entry = {}
            new_entry["hyp"] = decode_entry["hyp"] + [pi]
            #print(new_entry["hyp"])
            new_entry["score"] = decode_entry["score"] + pred_probs[pi]
            new_entry["dec_state"] = curr_dec_state
            new_entry["attn_v"] = ht

            new_entries.append(new_entry)

    # end with chainer test mode
    return new_entries


# In[24]:


def decode_beam(utt, curr_set, stop_limit=10, max_n=5, beam_width=3):
    with chainer.using_config('train', False):
        batch_data = get_utt_data(utt, curr_set)
        model.forward_enc(batch_data['X'])

        n_best = []

        if (len(batch_data['X']) > 0 and len(batch_data['y']) > 0):
            n_best.append(init_hyp())

            for i in range(stop_limit):
                #print("-"*40)
                #print(i)
                #print("-"*40)
                all_non_eos = [1 if e["hyp"][-1] != EOS_ID else 0 for e in n_best]
                if sum(all_non_eos) == 0:
                    #print("all eos at step={0:d}".format(i))
                    break

                curr_entries = []
                for e in n_best:
                    if e["hyp"][-1] != EOS_ID:
                        #print("feeding", v_dict["i2w"][e["hyp"][-1]])
                        curr_entries.extend(decode_beam_step(e, beam_width=beam_width))
                    else:
                        curr_entries.append(e)

                n_best = sorted(curr_entries, reverse=True, key=lambda t: t["score"])[:max_n]
    return n_best



all_valid_utts = [u for b in bucket_dict["fisher_dev"]["buckets"] for u in b]



utt_hyps = {}
for u in tqdm(all_valid_utts, ncols=80):
    with chainer.using_config('train', False):
        n_best = decode_beam(u, "fisher_dev", stop_limit=20, max_n=8, beam_width=3)
        utt_hyps[u] = [(e["hyp"], e["score"]) for e in n_best]


print("saving hyps")
pickle.dump(utt_hyps, open(os.path.join(m_cfg["model_dir"], "n_best_hyps.dict"), "wb"))


def clean_out_str(out_str):
    out_str = out_str.replace("`", "")
    out_str = out_str.replace('"', '')
    out_str = out_str.replace('¿', '')
    out_str = out_str.replace("''", "")
    out_str = out_str.strip()
    return out_str


# In[46]:


def get_out_str(h):
    out_str = ""
    for w in h:
        out_str += "{0:s}".format(w) if (w.startswith("'") or w=="n't") else " {0:s}".format(w)

    out_str = clean_out_str(out_str)
    return out_str


# In[47]:


MIN_LEN=0
MAX_LEN=300


# In[50]:


def write_to_file_len_filtered_preds(utts_beam, min_len, max_len):
    filt_utts = []
    for u in utts_beam:
        if (len(map_dict["fisher_dev"][u]["es_w"]) >= min_len and
           len(map_dict["fisher_dev"][u]["es_w"]) <= max_len):
            filt_utts.append(u)

    filt_utts = sorted(filt_utts)
    print("Utts matching len filter={0:d}".format(len(filt_utts)))

    hyp_path = os.path.join(m_cfg["model_dir"], "beam_min-{0:d}_max-{1:d}.en".format(min_len, max_len))
    print("writing hyps to: {0:s}".format(hyp_path))
    with open(hyp_path, "w") as out_f:
        for u in filt_utts:
            hyp = [v_dict['i2w'][i].decode() for i in utts_beam[u][0][0] if i >= 4]
            out_str = get_out_str(hyp)
            out_f.write("{0:s}\n".format(out_str))
    print("all done")


# In[51]:


print("writing to file")
write_to_file_len_filtered_preds(utt_hyps, 0, 300)

print("all done")

