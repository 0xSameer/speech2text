# coding: utf-8
from basics import *
from enc_dec import *
from prettytable import PrettyTable
import argparse
import textwrap
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, modified_precision
from nltk.translate.chrf_score import sentence_chrf, corpus_chrf
import copy
import nltk.translate.bleu_score

import fractions
import warnings
from collections import Counter

from stemming.porter2 import stem

from nltk.util import ngrams

try:
    fractions.Fraction(0, 1000, _normalize=False)
    from fractions import Fraction
except TypeError:
    from nltk.compat import Fraction

import prep_buckets

program_descrp = """run nmt experiments"""

'''
example:
python nmt_run.py -o $PWD/out -e 2 -k fisher_train
'''
xp = cuda.cupy

# -----------------------------------------------------------------------------
# helper functions for metrics
# -----------------------------------------------------------------------------
def get_en_words_from_list(l, join_str):
    ret_str = join_str.join([w.decode() for w in l])
    return ret_str.strip().split()

def basic_bleu(r, preds, dec_key, weights=(0.25, 0.25, 0.25, 0.25)):
    en_hyp = []
    en_ref = []

    if "bpe_w" == dec_key:
        ref_key = dec_key
    else:
        ref_key = 'en_w' if 'en_' in dec_key else 'es_w'

    join_str = ' ' if dec_key.endswith('_w') else ''

    total_matching_len = 0

    for p in preds:
        curr_p = []
        if type(p) == list:
            for i in p:
                if i == EOS_ID:
                    break
                else:
                    curr_p.append(i)
        en_hyp.append(curr_p)

    smooth_fun = nltk.translate.bleu_score.SmoothingFunction()

    b_score_value = corpus_bleu(r,
                          en_hyp,
                          weights=weights,
                          smoothing_function=smooth_fun.method2)

    return b_score_value, en_hyp

def calc_bleu(m_dict, v_dict, preds, utts, dec_key,
              weights=(0.25, 0.25, 0.25, 0.25),
              ref_index=-1):
    en_hyp = []
    en_ref = []

    if "bpe_w" == dec_key:
        ref_key = "en_c"
        join_str_ref = ""
    else:
        ref_key = 'en_w' if 'en_' in dec_key else 'es_w'
        join_str_ref = " "

    src_key = 'es_w'

    for u in tqdm(utts, ncols=80):
        if type(m_dict[u][ref_key]) == list:
            en_ref.append([get_en_words_from_list(m_dict[u][ref_key], join_str_ref)])
        else:
            if ref_index == -1:
                en_r_list = []
                for r in m_dict[u][ref_key]:
                    en_r_list.append(get_en_words_from_list(r, join_str_ref))
                en_ref.append(en_r_list)
            else:
                en_ref.append([get_en_words_from_list(m_dict[u][ref_key][ref_index], join_str_ref)])

    join_str = ' ' if dec_key.endswith('_w') else ''

    total_matching_len = 0

    for u, p in zip(utts, preds):
        total_matching_len += 1
        if type(p) == list:
            t_str = join_str.join([v_dict['i2w'][i].decode() for i in p])
            if dec_key == "bpe_w":
                t_str = t_str.replace("@@ ", "")
            t_str = t_str[:t_str.find('_EOS')]
            en_hyp.append(t_str.strip().split())
        else:
            en_hyp.append("")

    smooth_fun = nltk.translate.bleu_score.SmoothingFunction()

    b_score_value = corpus_bleu(en_ref,
                          en_hyp,
                          weights=weights,
                          smoothing_function=smooth_fun.method2)

    try:
        chrf_index = max(0, ref_index)
        chrf_score_value = corpus_chrf([r[chrf_index] for r in en_ref], en_hyp)
    except:
        chrf_score_value = 0

    return b_score_value, chrf_score_value, en_hyp, en_ref


def count_match(list1, list2):
    # each list can have repeated elements. The count should account for this.
    count1 = Counter(list1)
    count2 = Counter(list2)
    count1_keys = count1.keys()-set([UNK_ID, EOS_ID])
    count2_keys = count2.keys()-set([UNK_ID, EOS_ID])
    # count2_keys = count2.keys()
    common_w = set(count1_keys) & set(count2_keys)
    matches = sum([min(count1[w], count2[w]) for w in common_w])
    metrics = {}
    metrics["tc"] = {w: min(count1[w], count2[w]) for w in common_w}
    metrics["t"] = {w: count1[w] for w in count1_keys}
    metrics["tp"] = {w: count2[w] for w in count2_keys}

    tp = sum(metrics["tp"].values())
    t = sum(metrics["t"].values())

    return matches, tp, t, metrics

def basic_precision_recall(r, h, display=False):
    p_numerators = Counter() # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter() # Key = ngram order, and value = no. of ngram in ref.
    r_numerators = Counter() # Key = ngram order, and value = no. of ngram matches.
    r_denominators = Counter() # Key = ngram order, and value = no. of ngram in ref.
    metrics = {"rc": 0, "rt": 0, "tp": 0, "tc": 0, "word": {}}

    if display:
        print("total utts={0:d}".format(len(r)))

    i=1

    for references, hypothesis in zip(r, h):
        if min([len(any_ref) for any_ref in references]) > 0:
            if len(hypothesis) > 0:
                p_i = modified_precision(references, hypothesis, i)
                p_numerators[i] += p_i.numerator
                p_denominators[i] += p_i.denominator

                metrics["tc"] += p_i.numerator
                metrics["tp"] += p_i.denominator
            else:
                p_numerators[i] += 0
                p_denominators[i] += 0

                metrics["tc"] += 0
                metrics["tp"] += 0

            #print(p_i.numerator, p_i.denominator)

            tot_match = 0
            tot_count = 0

            max_recall_match, max_tp, max_t, max_word_level_details = count_match(references[0], hypothesis)
            max_recall = max_recall_match / max_t if max_t > 0 else 0

            for curr_ref in references:
                curr_match, curr_tp, curr_t, curr_word_level_details = count_match(curr_ref, hypothesis)
                curr_recall = curr_match / curr_t if curr_t > 0 else 0

                if curr_recall > max_recall:
                    max_recall_match = curr_match
                    max_t = curr_t
                    max_recall = curr_recall
                    max_word_level_details = curr_word_level_details

            r_numerators[i] += max_recall_match
            r_denominators[i] += max_t
            metrics["rc"] += max_recall_match
            metrics["rt"] += max_t
            for key in {"t","tp","tc"}:
                for w in max_word_level_details[key]:
                    if w not in metrics["word"]:
                        metrics["word"][w] = {"t": 0, "tp": 0, "tc": 0}
                    metrics["word"][w][key] += max_word_level_details[key][w]

    prec = [(n / d) * 100 if d > 0 else 0 for n,d in zip(p_numerators.values(), p_denominators.values())]
    rec = [(n / d) * 100 if d > 0 else 0 for n,d in zip(r_numerators.values(), r_denominators.values())]

    if display:
        print("{0:10s} | {1:>8s}".format("metric", "1-gram"))
        print("-"*54)
        print("{0:10s} | {1:8.2f}".format("precision", *prec))
        print("{0:10s} | {1:8.2f}".format("recall", *rec))

    return prec[0], rec[0], metrics

def corpus_precision_recall(r, h):
    p_numerators = Counter() # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter() # Key = ngram order, and value = no. of ngram in ref.
    r_numerators = Counter() # Key = ngram order, and value = no. of ngram matches.
    r_denominators = Counter() # Key = ngram order, and value = no. of ngram in ref.

    print("total utts={0:d}".format(len(r)))

    for references, hypothesis in zip(r, h):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate((0.25,.25,.25,.25), start=1):
            p_i, r_i = modified_precision_recall(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

            r_numerators[i] += r_i.numerator
            r_denominators[i] += r_i.denominator


    p = [(n / d) * 100 for n,d in zip(p_numerators.values(), p_denominators.values())]
    r = [(n / d) * 100 for n,d in zip(r_numerators.values(), r_denominators.values())]

    print("{0:10s} | {1:>8s} | {2:>8s}| {3:>8s} | {4:>8s}".format("metric", "1-gram","2-gram","3-gram","4-gram"))
    print("-"*54)
    print("{0:10s} | {1:8.2f} | {2:8.2f}| {3:8.2f} | {4:8.2f}".format("precision", *p))
    print("{0:10s} | {1:8.2f} | {2:8.2f}| {3:8.2f} | {4:8.2f}".format("recall", *r))


    return p, r

# def count_match(list1, list2):
#     # each list can have repeated elements. The count should account for this.
#     count1 = Counter(list1)
#     count2 = Counter(list2)
#     count2_keys = count2.keys()-set([UNK_ID, EOS_ID])
#     common_w = set(count1.keys()) & set(count2_keys)
#     matches = sum([min(count1[w], count2[w]) for w in common_w])
#     return matches

def modified_precision_recall(references, hypothesis, n):
    # Extracts all ngrams in hypothesis
    # Set an empty Counter if hypothesis is empty.
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    # Extract a union of references' counts.
    ## max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
    max_counts = {}
    max_reference_count = 0
    total_ref_count = 0
    for reference in references:
        reference_counts = Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        for ngram in reference_counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0),
                                    reference_counts[ngram])
        ref_length = sum(reference_counts.values())
        if ref_length > max_reference_count:
            max_reference_count = ref_length
        total_ref_count += ref_length

    # Assigns the intersection between hypothesis and references' counts.
    clipped_counts = {ngram: min(count, max_counts.get(ngram, 0))
                      for ngram, count in counts.items()}

    numerator = sum(clipped_counts.values())
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, sum(counts.values()))
    rec_denominator = max(1, sum(max_counts.values()))

    prec = Fraction(numerator, denominator, _normalize=False)
    rec = Fraction(numerator, rec_denominator, _normalize=False)

    return prec, rec


def get_batch(m_dict, x_key, y_key, utt_list, vocab_dict,
              max_enc, max_dec, input_path='', limit_vocab=False, add_unk=False):
    batch_data = {'X':[], 'y':[], 'r':[], "utts": []}
    # -------------------------------------------------------------------------
    # loop through each utterance in utt list
    # -------------------------------------------------------------------------
    for u in utt_list:
        # ---------------------------------------------------------------------
        #  add X data
        # ---------------------------------------------------------------------
        if x_key == 'sp':
            # -----------------------------------------------------------------
            # for speech data
            # -----------------------------------------------------------------
            # get path to speech file
            utt_sp_path = os.path.join(input_path, "{0:s}.npy".format(u))
            if not os.path.exists(utt_sp_path):
                # for training data, there are sub-folders
                utt_sp_path = os.path.join(input_path,
                                           u.split('_',1)[0],
                                           "{0:s}.npy".format(u))
            if os.path.exists(utt_sp_path):
                x_data = xp.load(utt_sp_path)[:max_enc]
            else:
                # -------------------------------------------------------------
                # exception if file not found
                # -------------------------------------------------------------
                raise FileNotFoundError("ERROR!! file not found: {0:s}".format(utt_sp_path))
                # -------------------------------------------------------------
        else:
            # -----------------------------------------------------------------
            # for text data
            # -----------------------------------------------------------------
            x_ids = [vocab_dict[x_key]['w2i'].get(w, UNK_ID) for w in m_dict[u][x_key]]
            x_data = xp.asarray(x_ids, dtype=xp.int32)[:max_enc]
            # -----------------------------------------------------------------
        # ---------------------------------------------------------------------
        #  add labels
        # ---------------------------------------------------------------------
        if limit_vocab == False:
            if type(m_dict[u][y_key]) == list:
                en_ids = [vocab_dict[y_key]['w2i'].get(w, UNK_ID) for w in m_dict[u][y_key]]
                r_data = [en_ids[:max_dec]]
            else:
                # dev and test data have multiple translations
                # choose the first one for computing perplexity
                en_ids = [vocab_dict[y_key]['w2i'].get(w, UNK_ID) for w in m_dict[u][y_key][0]]
                r_data = []
                for r in m_dict[u][y_key]:
                    r_list = [vocab_dict[y_key]['w2i'].get(w, UNK_ID) for w in r]
                    r_data.append(r_list[:max_dec])
        else:
            if type(m_dict[u][y_key]) == list:
                en_ids = []
                for w in m_dict[u][y_key]:
                    if w in vocab_dict['w2i']:
                        en_ids.append(vocab_dict['w2i'][w])
                    # end if
                # end for
                # Don't add unk for r_data
                r_data = [en_ids[:max_dec]]
                if len(en_ids) == 0 and add_unk:
                    en_ids = [UNK_ID]
            else:
                # dev and test data have multiple translations
                # choose the first one for computing perplexity
                en_ids = []
                for w in m_dict[u][y_key][0]:
                    if w in vocab_dict['w2i']:
                        en_ids.append(vocab_dict['w2i'][w])
                    # end if
                # end for
                if len(en_ids) == 0 and add_unk:
                    en_ids = [UNK_ID]

                r_data = []
                for r in m_dict[u][y_key]:
                    r_list = []
                    for w in r:
                        if w in vocab_dict['w2i']:
                            r_list.append(vocab_dict['w2i'][w])
                        # end if
                    # end for
                    # if len(r_list) == 0 and add_unk:
                    #     r_list = [UNK_ID]
                    r_data.append(r_list[:max_dec])
                # end for
            # end else
        y_ids = [GO_ID] + en_ids[:max_dec-2] + [EOS_ID]
        # ---------------------------------------------------------------------
        if len(x_data) > 0 and len(en_ids) > 0:
            batch_data['X'].append(x_data)
            batch_data['y'].append(xp.asarray(y_ids, dtype=xp.int32))
            batch_data['r'].append(r_data)
            batch_data['utts'].append(u)
    # -------------------------------------------------------------------------
    # end for all utterances in batch
    # -------------------------------------------------------------------------
    if len(batch_data['X']) > 0 and len(batch_data['y']) > 0:
        batch_data['X'] = F.pad_sequence(batch_data['X'], padding=PAD_ID)
        batch_data['y'] = F.pad_sequence(batch_data['y'], padding=PAD_ID)
    return batch_data

def create_batches(b_dict, batch_size):
    num_b = b_dict['num_b']
    width_b = b_dict['width_b']
    b_shuffled = list(range(num_b))
    random.shuffle(b_shuffled)
    total_utts = 0
    utt_list_batches = []
    # 'max': 256, 'med': 200, 'min': 100, 'scale':1
    for b in b_shuffled:
        # ---------------------------------------------------------------------
        # compute batch size to use for current bucket
        # ---------------------------------------------------------------------
        if b < num_b // 3:
            b_size = int(batch_size['max'])
        elif b < (num_b*2) // 3:
            b_size = int(batch_size['med'])
        else:
            b_size = int(batch_size['min'])
        # ---------------------------------------------------------------------
        # old logic: divide each by batch_size['scale']
        # ---------------------------------------------------------------------
        bucket = b_dict['buckets'][b]
        b_len = len(bucket)
        total_utts += b_len
        random.shuffle(bucket)
        # ---------------------------------------------------------------------
        # append all utterances in slices of batch size
        # ---------------------------------------------------------------------
        for i in range(0,b_len, b_size):
            utt_list_batches.append((bucket[i:i+b_size],b))
        # ---------------------------------------------------------------------
        # end bucket loop
    # end all buckets loop
    # -------------------------------------------------------------------------
    # shuffle the entire list of batches
    random.shuffle(utt_list_batches)
    return utt_list_batches, total_utts

def feed_model(model, optimizer, m_dict, b_dict,
               batch_size, vocab_dict, x_key, y_key,
               train, input_path, max_dec, t_cfg,
               use_y=True, limit_vocab=False, add_unk=False):
    # number of buckets
    num_b = b_dict['num_b']
    width_b = b_dict['width_b']
    pred_sents = []
    ref_sents = []
    utts = []
    total_loss = 0
    loss_per_epoch = 0
    total_loss_updates= 0

    sys.stderr.flush()
    # -------------------------------------------------------------------------
    # create batches of utterances - shuffled
    # -------------------------------------------------------------------------
    utt_list_batches, total_utts = create_batches(b_dict, batch_size)
    # -------------------------------------------------------------------------
    with tqdm(total=total_utts, ncols=80) as pbar:
        for i, (utt_list, b) in enumerate(utt_list_batches):
            # -----------------------------------------------------------------
            # get batch_data
            # -----------------------------------------------------------------
            batch_data = get_batch(m_dict,
                                   x_key, y_key,
                                   utt_list,
                                   vocab_dict,
                                   ((b+1) * width_b),
                                   max_dec,
                                   input_path=input_path,
                                   limit_vocab=limit_vocab,
                                   add_unk=add_unk)
            # -----------------------------------------------------------------
            if (len(batch_data['X']) > 0 and len(batch_data['y']) > 0):
                if use_y:
                    # ---------------------------------------------------------
                    # using labels, computing loss
                    # also used for dev set
                    # ---------------------------------------------------------
                    with chainer.using_config('train', train):
                        cuda.get_device(t_cfg['gpuid']).use()
                        p, loss = model.forward(X=batch_data['X'],
                                    y=batch_data['y'],
                                    add_noise=t_cfg['speech_noise'],
                                    teacher_ratio = t_cfg['teach_ratio'])
                        loss_val = float(loss.data) / batch_data['y'].shape[1]
                else:
                    # ---------------------------------------------------------
                    # prediction only
                    # ---------------------------------------------------------
                    with chainer.using_config('train', False):
                        cuda.get_device(t_cfg['gpuid']).use()
                        p, _ = model.forward(X=batch_data['X'])
                        loss_val = 0.0
                # -------------------------------------------------------------
                # add list of utterances used
                # -------------------------------------------------------------
                #utts.extend(utt_list)
                utts.extend(batch_data['utts'])
                # -------------------------------------------------------------
                if len(p) > 0:
                    p_list = p.tolist()
                    pred_sents.extend(p_list)
                    ref_sents.extend(batch_data['r'])

                total_loss += loss_val
                total_loss_updates += 1
                loss_per_epoch = (total_loss / total_loss_updates)

                out_str = "b={0:d},l={1:.2f},avg={2:.2f}".format((b+1),loss_val,loss_per_epoch)
                # -------------------------------------------------------------
                # train mode logic
                # -------------------------------------------------------------
                if train:
                    # if SGD, apply linear scaling for learning rate:
                    # https://arxiv.org/pdf/1706.02677.pdf
                    if 'lr_scale' in t_cfg and t_cfg['lr_scale'] == True and t_cfg['optimizer'] == OPT_SGD:
                        if len(utt_list) > batch_size['min']:
                            lr_scaled = t_cfg['lr'] * (len(utt_list) / batch_size['min'])
                            optimizer.hyperparam.lr = lr_scaled
                        else:
                            optimizer.hyperparam.lr = t_cfg['lr']

                        out_str = "b={0:d},l={1:.2f},avg={2:.2f},lr={3:.7f}".format((b+1),loss_val,loss_per_epoch, optimizer.hyperparam.lr)
                    # ---------------------------------------------------------
                    model.cleargrads()
                    loss.backward()
                    optimizer.update()
                    # ---------------------------------------------------------

                pbar.set_description('{0:s}'.format(out_str))
            else:
                print("no data in batch")
                print(utt_list)
            # update progress bar
            #pbar.update(len(utt_list))
            pbar.update(len(batch_data['utts']))
        # end for batches
    # end tqdm
    return pred_sents, ref_sents, utts, loss_per_epoch
# end feed_model

# map_dict, vocab_dict, bucket_dict = get_data_dicts(model_cfg)
def get_data_dicts(m_cfg):
    print("-"*50)
    # load dictionaries
    # -------------------------------------------------------------------------
    # MAP dict
    # -------------------------------------------------------------------------
    v_pre = m_cfg['vocab_pre']
    map_dict_path = os.path.join(m_cfg['data_path'], v_pre+'map.dict')
    print("loading dict: {0:s}".format(map_dict_path))
    map_dict = pickle.load(open(map_dict_path, "rb"))
    # -------------------------------------------------------------------------
    # VOCAB
    # -------------------------------------------------------------------------
    if 'limit_vocab' in m_cfg and m_cfg["limit_vocab"] == True:
        vocab_path = os.path.join(m_cfg['data_path'], m_cfg["vocab_path"])
    else:
        if 'fisher' in m_cfg['train_set']:
            if m_cfg['stemmify'] == False:
                vocab_path = os.path.join(m_cfg['data_path'], v_pre+'train_vocab.dict')
            else:
                vocab_path = os.path.join(m_cfg['data_path'], v_pre+'train_stemmed_vocab.dict')
        else:
            vocab_path = os.path.join(m_cfg['data_path'], v_pre+'ch_train_vocab.dict')
    print("loading dict: {0:s}".format(vocab_path))
    vocab_dict = pickle.load(open(vocab_path, "rb"))
    print("-"*50)
    # -------------------------------------------------------------------------
    # BUCKETS
    # -------------------------------------------------------------------------
    prep_buckets.buckets_main(m_cfg['data_path'],
                              m_cfg['buckets_num'],
                              m_cfg['buckets_width'],
                              m_cfg['enc_key'],
                              scale=m_cfg['train_scale'],
                              seed=m_cfg['seed'])

    buckets_path = os.path.join(m_cfg['data_path'],
                                'buckets_{0:s}.dict'.format(m_cfg['enc_key']))
    print("loading dict: {0:s}".format(buckets_path))
    bucket_dict = pickle.load(open(buckets_path, "rb"))
    print("-"*50)
    # -------------------------------------------------------------------------
    # INFORMATION
    # -------------------------------------------------------------------------
    for cat in map_dict:
        print('utterances in {0:s} = {1:d}'.format(cat, len(map_dict[cat])))

    if m_cfg['enc_key'] != 'sp':
        vocab_size_es = len(vocab_dict[m_cfg['enc_key']]['w2i'])
    else:
        vocab_size_es = 0

    if 'limit_vocab' in m_cfg and m_cfg["limit_vocab"] == True:
        vocab_size_en = len(vocab_dict['w2i'])
    else:
        vocab_size_en = len(vocab_dict[m_cfg['dec_key']]['w2i'])
    print('vocab size for {0:s} = {1:d}'.format(m_cfg['enc_key'],
                                                vocab_size_es))
    print('vocab size for {0:s} = {1:d}'.format(m_cfg['dec_key'],
                                                vocab_size_en))
    # -------------------------------------------------------------------------
    return map_dict, vocab_dict, bucket_dict

def check_model(cfg_path):
    # -------------------------------------------------------------------------
    # read config files model
    # -------------------------------------------------------------------------
    with open(os.path.join(cfg_path, "model_cfg.json"), "r") as model_f:
        m_cfg = json.load(model_f)
    # -------------------------------------------------------------------------
    with open(os.path.join(cfg_path, "train_cfg.json"), "r") as train_f:
        t_cfg = json.load(train_f)
    xp = cuda.cupy if t_cfg['gpuid'] >= 0 else np
    # -------------------------------------------------------------------------
    # check model path
    # -------------------------------------------------------------------------
    if not os.path.exists(m_cfg['data_path']):
        raise FileNotFoundError("ERROR!! file not found: {0:s}".format(m_cfg['data_path']))
    # end if
    # -------------------------------------------------------------------------
    # initialize new model
    # -------------------------------------------------------------------------
    model = SpeechEncoderDecoder(m_cfg, t_cfg['gpuid'])
    model.to_gpu(t_cfg['gpuid'])
    # -------------------------------------------------------------------------
    # set up optimizer
    # -------------------------------------------------------------------------
    if t_cfg['optimizer'] == OPT_ADAM:
        print("using ADAM optimizer")
        optimizer = optimizers.Adam(alpha=t_cfg['lr'],
                                    beta1=0.9,
                                    beta2=0.999,
                                    eps=1e-08)
    else:
        print("using SGD optimizer")
        optimizer = optimizers.SGD(lr=t_cfg['lr'])

    # attach optimizer
    optimizer.setup(model)
    # -------------------------------------------------------------------------
    # optimizer settings
    # -------------------------------------------------------------------------
    if m_cfg['l2'] > 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(m_cfg['l2']))

    # gradient clipping
    optimizer.add_hook(chainer.optimizer.GradientClipping(threshold=m_cfg['grad_clip']))

    # gradient noise
    if t_cfg['grad_noise_eta'] > 0:
        print("------ Adding gradient noise")
        optimizer.add_hook(chainer.optimizer.GradientNoise(eta=t_cfg['grad_noise_eta']))
        print("Finished adding gradient noise")
    # -------------------------------------------------------------------------
    # check last saved model
    # -------------------------------------------------------------------------
    max_epoch = 0
    # -------------------------------------------------------------------------
    # add debug info
    # -------------------------------------------------------------------------
    m_cfg['model_dir'] = cfg_path
    m_cfg['train_log'] = os.path.join(m_cfg['model_dir'], "train.log")
    m_cfg['dev_log'] = os.path.join(m_cfg['model_dir'], "dev.log")
    m_cfg['model_fname'] = os.path.join(m_cfg['model_dir'], "seq2seq.model")
    m_cfg['opt_fname'] = os.path.join(m_cfg['model_dir'], "train.opt")
    # -------------------------------------------------------------------------
    model_fil = m_cfg['model_fname']
    model_files = [f for f in os.listdir(os.path.dirname(model_fil))
                   if os.path.basename(model_fil).replace('.model','') in f]
    if len(model_files) > 0:
        print("-"*80)
        max_model_fil = max(model_files, key=lambda s: int(s.split('_')[-1].split('.')[0]))
        max_model_fil = os.path.join(os.path.dirname(model_fil),
                                     max_model_fil)
        print('model found = \n{0:s}'.format(max_model_fil))
        serializers.load_npz(max_model_fil, model)
        print("finished loading ..")
        max_epoch = int(max_model_fil.split('_')[-1].split('.')[0])
        # load optimizer
        if os.path.exists(m_cfg['opt_fname']):
            print("optimizer found = {0:s}".format(m_cfg['opt_fname']))
            serializers.load_npz(m_cfg['opt_fname'], optimizer)
            print("finished loading optimizer ...")
        else:
            print("optimizer not found")
    else:
        print("-"*80)
        print('model not found')
    # end if model found
    # -------------------------------------------------------------------------
    return max_epoch, model, optimizer, m_cfg, t_cfg
# end check_model

def train_loop(cfg_path, epochs):
    # -------------------------------------------------------------------------
    # check for existing model
    # -------------------------------------------------------------------------
    last_epoch, model, optimizer, m_cfg, t_cfg = check_model(cfg_path)
    # -------------------------------------------------------------------------
    train_key = m_cfg['train_set']
    dev_key = m_cfg['dev_set']
    batch_size=t_cfg['batch_size']
    enc_key=m_cfg['enc_key']
    dec_key=m_cfg['dec_key']
    limit_vocab = False if 'limit_vocab' not in m_cfg else m_cfg['limit_vocab']
    add_unk = False if 'add_unk' not in m_cfg else m_cfg['add_unk']
    # -------------------------------------------------------------------------
    # get data dictionaries
    # -------------------------------------------------------------------------
    map_dict, vocab_dict, bucket_dict = get_data_dicts(m_cfg)
    # -------------------------------------------------------------------------
    # start train loop
    # -------------------------------------------------------------------------
    with open(m_cfg['train_log'], mode='a') as train_log, open(m_cfg['dev_log'], mode='a') as dev_log:
        for i in range(epochs):
            print("-"*80)
            print("EPOCH = {0:d} / {1:d}".format(last_epoch+i+1, last_epoch+epochs))
            print("using GPU={0:d}".format(t_cfg['gpuid']))
            print('model details in : {0:s}'.format(m_cfg['model_dir']))
            # -----------------------------------------------------------------
            # Check to add Gaussian weight noise
            # -----------------------------------------------------------------
            if (last_epoch+i+1 >= t_cfg['iter_weight_noise']) and (t_cfg['iter_weight_noise'] > 0):
                print("Adding Gaussian weight noise, mean={0:.2f}, stdev={1:0.6f}".format(t_cfg['weight_noise_mean'], t_cfg['weight_noise_sigma']))
                model.add_weight_noise(t_cfg['weight_noise_mean'], t_cfg['weight_noise_sigma'])
                print("Finished adding Gaussian weight noise")
            # end adding gaussian weight noise
            # -----------------------------------------------------------------
            # train
            # -----------------------------------------------------------------
            input_path = os.path.join(m_cfg['data_path'],
                                      m_cfg['train_set'])
            pred_sents, ref_sents, utts, train_loss = feed_model(model,
                                              optimizer=optimizer,
                                              m_dict=map_dict[train_key],
                                              b_dict=bucket_dict[train_key],
                                              vocab_dict=vocab_dict,
                                              batch_size=batch_size,
                                              x_key=enc_key,
                                              y_key=dec_key,
                                              train=True,
                                              input_path=input_path,
                                              max_dec=m_cfg['max_en_pred'],
                                              t_cfg=t_cfg,
                                              use_y=True,
                                              limit_vocab=limit_vocab,
                                              add_unk=add_unk)
            # log train loss
            train_log.write("{0:d}, {1:.4f}\n".format(last_epoch+i+1, train_loss))
            train_log.flush()
            os.fsync(train_log.fileno())
            # -----------------------------------------------------------------

            # -----------------------------------------------------------------
            # dev
            # -----------------------------------------------------------------
            input_path = os.path.join(m_cfg['data_path'],
                                      m_cfg['dev_set'])
            pred_sents, ref_sents, utts, dev_loss = feed_model(model,
                                              optimizer=optimizer,
                                              m_dict=map_dict[dev_key],
                                              b_dict=bucket_dict[dev_key],
                                              vocab_dict=vocab_dict,
                                              batch_size=batch_size,
                                              x_key=enc_key,
                                              y_key=dec_key,
                                              train=False,
                                              input_path=input_path,
                                              max_dec=m_cfg['max_en_pred'],
                                              t_cfg=t_cfg,
                                              use_y=True,
                                              limit_vocab=limit_vocab,
                                              add_unk=add_unk)

            smooth_fun = nltk.translate.bleu_score.SmoothingFunction()

            if dec_key.endswith("_w"):
                # dev_b_score, dev_preds = basic_bleu(ref_sents, pred_sents, dec_key)
                # dev_prec, dev_rec, _ = basic_precision_recall(ref_sents, dev_preds)

                dev_b_score, _, char_h, char_r = calc_bleu(map_dict[dev_key],
                                                           vocab_dict[dec_key],
                                                           pred_sents, utts,
                                                           dec_key)
                dev_prec, dev_rec, _ = basic_precision_recall(char_r, char_h)
            else:
                # dev_b_score = corpus_bleu(ref_sents,
                #                           pred_sents,
                #                           weights=[0.25, 0.25, 0.25, 0.25],
                #                           smoothing_function=smooth_fun.method2)
                dev_b_score, _, char_h, char_r = calc_bleu(map_dict[dev_key],
                                                           vocab_dict[dec_key],
                                                           pred_sents, utts,
                                                           dec_key)
                dev_prec, dev_rec, _ = basic_precision_recall(char_r, char_h)

            # log dev loss
            dev_log.write("{0:d}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}\n".format(last_epoch+i+1, dev_loss, dev_b_score, dev_prec, dev_rec))
            dev_log.flush()
            os.fsync(dev_log.fileno())

            print("^"*80)
            print("{0:s} train avg loss={1:.4f}, dev avg loss={2:.4f}, dev bleu={3:.4f}".format("*" * 10, train_loss, dev_loss, dev_b_score))
            print("{0:s} dev: prec={1:.3f}, recall={2:.3f}".format("*" * 10, dev_prec, dev_rec))
            print("^"*80)
            # -----------------------------------------------------------------
            # save model
            # -----------------------------------------------------------------
            model_fil = m_cfg['model_fname']
            if ((i+1) % t_cfg['iters_save_model'] == 0) or (i == (epochs-1)):
                print("Saving model")
                serializers.save_npz(model_fil.replace(".model", "_{0:d}.model".format(last_epoch+i+1)), model)
                print("Finished saving model")
                print("Saving optimizer")
                serializers.save_npz(m_cfg['opt_fname'], optimizer)
                print("Finished saving optimizer")
            # end if save model
            # -----------------------------------------------------------------
        # end for epochs
    # end open log files
# end train loop

# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=program_descrp)
    parser.add_argument('-m','--cfg_path', help='path for model config',
                        required=True)
    parser.add_argument('-e','--epochs', help='num epochs',
                        required=True)

    args = vars(parser.parse_args())

    cfg_path = args['cfg_path']

    epochs = int(args['epochs'])

    print("number of epochs={0:d}".format(epochs))

    # -------------------------------------------------------------------------
    # call train loop
    # -------------------------------------------------------------------------
    train_loop(cfg_path=cfg_path, epochs=epochs)
    # -------------------------------------------------------------------------
    print("all done ...")
# end main
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
# -----------------------------------------------------------------------------
