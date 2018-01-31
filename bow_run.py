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
def get_en_words_from_list(l):
    return [w.decode() for w in l]


def count_match(list1, list2):
    # each list can have repeated elements. The count should account for this.
    count1 = Counter(list1)
    count2 = Counter(list2)
    # count2_keys = count2.keys()-set([UNK_ID, EOS_ID])
    count2_keys = count2.keys()
    common_w = set(count1.keys()) & set(count2_keys)
    matches = sum([min(count1[w], count2[w]) for w in common_w])
    return matches

def basic_precision_recall(r, h, display=False):
    p_numerators = Counter() # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter() # Key = ngram order, and value = no. of ngram in ref.
    r_numerators = Counter() # Key = ngram order, and value = no. of ngram matches.
    r_denominators = Counter() # Key = ngram order, and value = no. of ngram in ref.
    metrics = {"rc": 0, "rt": 0, "tp": 0, "tc": 0}

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


            max_recall_match = count_match(references[0], hypothesis)
            max_recall_count = len(references[0])
            max_recall = max_recall_match / max_recall_count if max_recall_count > 0 else 0

            for curr_ref in references:
                curr_match = count_match(curr_ref, hypothesis)

                curr_count = len(curr_ref)
                curr_recall = curr_match / curr_count if curr_count > 0 else 0

                if curr_recall > max_recall:
                    max_recall_match = curr_match
                    max_recall_count = curr_count
                    max_recall = curr_recall

            r_numerators[i] += max_recall_match
            r_denominators[i] += max_recall_count
            metrics["rc"] += max_recall_match
            metrics["rt"] += max_recall_count

    prec = [(n / d) * 100 if d > 0 else 0 for n,d in zip(p_numerators.values(), p_denominators.values())]
    rec = [(n / d) * 100 if d > 0 else 0 for n,d in zip(r_numerators.values(), r_denominators.values())]

    if display:
        print("{0:10s} | {1:>8s}".format("metric", "1-gram"))
        print("-"*54)
        print("{0:10s} | {1:8.2f}".format("precision", *prec))
        print("{0:10s} | {1:8.2f}".format("recall", *rec))

    return prec[0], rec[0], metrics


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


def get_pred_words_from_probs(probs, thresh_vals, pred_limit):
    pred_words = []
    for row in probs:
        pred_inds = np.where(row >= thresh_vals)[0]
        if len(pred_inds) > pred_limit:
            pred_inds = np.argsort(row)[-pred_limit:][::-1]
        pred_words.append([i for i in pred_inds.tolist() if i > 3])
    return pred_words

def get_bow_batch(m_dict, x_key, y_key, utt_list, vocab_dict, bow_dict,
                  max_enc, max_dec, input_path=''):
    batch_data = {'X':[], 't':[], 'y':[], 'r':[], 'l': []}
    # -------------------------------------------------------------------------
    # loop through each utterance in utt list
    # -------------------------------------------------------------------------
    for i, u in enumerate(utt_list):
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
        if type(m_dict[u][y_key]) == list:
            en_ids = list(set([bow_dict['w2i'].get(w, UNK_ID) for w in m_dict[u][y_key]])-set(range(4)))
            r_data = [en_ids[:max_dec]]

        else:
            # dev and test data have multiple translations
            # choose the first one for computing perplexity
            en_ids = list(set([bow_dict['w2i'].get(w, UNK_ID) for w in m_dict[u][y_key][0]])-set(range(4)))
            r_data = []
            for r in m_dict[u][y_key]:
                r_list = list(set([bow_dict['w2i'].get(w, UNK_ID) for w in r])-set(range(4)))
                r_data.append(r_list[:max_dec])

        y_ids = en_ids[:max_dec]
        # ---------------------------------------------------------------------
        if len(x_data) > 0:
            #  and len(y_ids) > 0
            batch_data['X'].append(x_data)
            batch_data['t'].append([y_ids])
            y_data = xp.zeros(len(bow_dict['w2i']), dtype=xp.int32)
            y_data[y_ids] = 1
            y_data[list(range(4))] = -1
            batch_data['y'].append(y_data)
            batch_data['r'].append(r_data)
            batch_data['l'].append(len(x_data))

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
               batch_size, vocab_dict, bow_dict, x_key, y_key,
               train, input_path, max_dec, t_cfg, use_y=True):
    # number of buckets
    num_b = b_dict['num_b']
    width_b = b_dict['width_b']
    utts = {"ids": [], "preds": [], "probs": [], "refs": []}

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
            batch_data = get_bow_batch(m_dict,
                                   x_key, y_key,
                                   utt_list,
                                   vocab_dict,
                                   bow_dict,
                                   ((b+1) * width_b),
                                   max_dec,
                                   input_path=input_path)
            # -----------------------------------------------------------------
            if (len(batch_data['X']) > 0 and len(batch_data['y']) > 0):
                if use_y:
                    # ---------------------------------------------------------
                    # using labels, computing loss
                    # also used for dev set
                    # ---------------------------------------------------------
                    with chainer.using_config('train', train):
                        cuda.get_device(t_cfg['gpuid']).use()
                        p_words, loss, p_probs = model.forward_bow(X=batch_data['X'],
                                                                   y=batch_data['y'],
                                                                   add_noise=t_cfg['speech_noise'],
                                                                   l=batch_data['l'])
                        loss_val = float(loss.data)
                else:
                    # ---------------------------------------------------------
                    # prediction only
                    # ---------------------------------------------------------
                    with chainer.using_config('train', False):
                        cuda.get_device(t_cfg['gpuid']).use()
                        p_words, _, p_probs = model.forward_bow(X=batch_data['X'], l=batch_data['l'])
                        loss_val = 0.0
                # -------------------------------------------------------------
                # add list of utterances used
                # -------------------------------------------------------------
                for u, pred, prob, ref in zip(utt_list, p_words, p_probs, batch_data['r']):
                    utts['ids'].append(u)
                    utts["preds"].append(pred)
                    utts["probs"].append(prob)
                    utts["refs"].append(ref)
                # utts.extend(utt_list)
                # -------------------------------------------------------------
                # if len(p) > 0:
                #     pred_sents.extend(p)
                #     refs.extend(batch_data['t'])

                total_loss += loss_val
                total_loss_updates += 1
                loss_per_epoch = (total_loss / total_loss_updates)

                out_str = "b={0:d},l={1:.2f},avg={2:.2f}".format((b+1),loss_val,loss_per_epoch)
                # -------------------------------------------------------------
                # train mode logic
                # -------------------------------------------------------------
                if train:
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
            pbar.update(len(utt_list))
        # end for batches
    # end tqdm
    # return pred_sents, utts, refs, loss_per_epoch
    utts["probs"] = F.sigmoid(F.pad_sequence(utts["probs"]))
    utts["probs"].to_cpu()
    utts["probs"] = utts["probs"].data

    return utts, loss_per_epoch
# end feed_model

# map_dict, vocab_dict, bucket_dict = get_data_dicts(model_cfg)
def get_data_dicts(m_cfg):
    print("-"*50)
    # load dictionaries
    # -------------------------------------------------------------------------
    # MAP dict
    # -------------------------------------------------------------------------
    map_dict_path = os.path.join(m_cfg['data_path'],'map.dict')
    print("loading dict: {0:s}".format(map_dict_path))
    map_dict = pickle.load(open(map_dict_path, "rb"))
    # -------------------------------------------------------------------------
    # VOCAB
    # -------------------------------------------------------------------------
    if 'fisher' in m_cfg['train_set']:
        if m_cfg['stemmify'] == False:
            vocab_path = os.path.join(m_cfg['data_path'], 'train_vocab.dict')
        else:
            vocab_path = os.path.join(m_cfg['data_path'], 'train_stemmed_vocab.dict')
    else:
        vocab_path = os.path.join(m_cfg['data_path'], 'ch_train_vocab.dict')
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
    # bag-of-words
    # -------------------------------------------------------------------------
    bow_dict_path = os.path.join(m_cfg['data_path'], m_cfg['bagofwords_vocab'])
    print("loading dict: {0:s}".format(bow_dict_path))
    bow_dict = pickle.load(open(bow_dict_path, "rb"))
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
    vocab_size_en = len(vocab_dict[m_cfg['dec_key']]['w2i'])
    print('vocab size for {0:s} = {1:d}'.format(m_cfg['enc_key'],
                                                vocab_size_es))
    print('vocab size for {0:s} = {1:d}'.format(m_cfg['dec_key'],
                                                vocab_size_en))
    # -------------------------------------------------------------------------
    return map_dict, vocab_dict, bucket_dict, bow_dict

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
    # -------------------------------------------------------------------------
    # get data dictionaries
    # -------------------------------------------------------------------------
    map_dict, vocab_dict, bucket_dict, bow_dict = get_data_dicts(m_cfg)
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
            train_utts, train_loss = feed_model(model,
                                          optimizer=optimizer,
                                          m_dict=map_dict[train_key],
                                          b_dict=bucket_dict[train_key],
                                          vocab_dict=vocab_dict,
                                          bow_dict=bow_dict,
                                          batch_size=batch_size,
                                          x_key=enc_key,
                                          y_key=dec_key,
                                          train=True,
                                          input_path=input_path,
                                          max_dec=m_cfg['max_en_pred'],
                                          t_cfg=t_cfg,
                                          use_y=True)

            mean_pos_scores = np.array([0.0 for _ in bow_dict["i2w"]], dtype="f")
            mean_neg_scores = np.array([0.0 for _ in bow_dict["i2w"]], dtype="f")


            for i_w in range(4, len(bow_dict["i2w"])):
                this_word = bow_dict["i2w"][i_w]
                pos_indx = [i_w in r[0] for r in train_utts["refs"]]
                neg_indx = [i_w not in r[0] for r in train_utts["refs"]]
                mean_pos_scores[i_w] = np.mean(F.sigmoid(train_utts["probs"][:,i_w][pos_indx]).data)
                mean_neg_scores[i_w] = np.mean(F.sigmoid(train_utts["probs"][:,i_w][neg_indx]).data)

            train_pred_words = get_pred_words_from_probs(train_utts["probs"],
                                                       m_cfg["pred_thresh"],
                                                       m_cfg['max_en_pred'])

            train_prec, train_rec, _ = basic_precision_recall(train_utts["refs"], train_pred_words)
            # log train loss
            train_log.write("{0:d}, {1:.4f}, {2:.4f}, {3:.4f}\n".format(last_epoch+i+1,
                                                                 train_loss,
                                                                 train_prec,
                                                                 train_rec))
            train_log.flush()
            os.fsync(train_log.fileno())
            # -----------------------------------------------------------------

            # -----------------------------------------------------------------
            # dev
            # -----------------------------------------------------------------
            input_path = os.path.join(m_cfg['data_path'],
                                      m_cfg['dev_set'])
            dev_utts, dev_loss = feed_model(model,
                                        optimizer=optimizer,
                                        m_dict=map_dict[dev_key],
                                        b_dict=bucket_dict[dev_key],
                                        vocab_dict=vocab_dict,
                                        bow_dict=bow_dict,
                                        batch_size=batch_size,
                                        x_key=enc_key,
                                        y_key=dec_key,
                                        train=False,
                                        input_path=input_path,
                                        max_dec=m_cfg['max_en_pred'],
                                        t_cfg=t_cfg,
                                        use_y=True)

            dev_pred_words = get_pred_words_from_probs(dev_utts["probs"],
                                                       m_cfg["pred_thresh"],
                                                       m_cfg['max_en_pred'])

            prec, rec, _ = basic_precision_recall(dev_utts["refs"], dev_pred_words)

            # log dev loss
            dev_log.write("{0:d}, {1:.4f}, {2:.4f}, {3:.4f}\n".format(last_epoch+i+1, dev_loss, prec, rec))
            dev_log.flush()
            os.fsync(dev_log.fileno())

            print("^"*80)
            print("{0:s} train avg loss={1:.4f}, dev avg loss={2:.4f}".format("*" * 10, train_loss, dev_loss))
            print("^"*80)
            print("{0:s} train: prec={1:.3f}, recall={2:.3f} ----- dev: prec={3:.3f}, recall={4:.3f}".format("*" * 10, train_prec, train_rec, prec, rec))
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
                print("Saving utterance predictions")
            # else:
            #     print("Saving model")
            #     serializers.save_npz(model_fil.replace(".model", "_last.model", model))
            #     print("Finished saving model")
            pickle.dump(dev_utts, open(os.path.join(m_cfg['model_dir'], "model_s2t_dev_out.dict"), "wb"))
            pickle.dump(mean_pos_scores, open(os.path.join(m_cfg['model_dir'], "mean_pos_scores.dict"), "wb"))
            pickle.dump(mean_neg_scores, open(os.path.join(m_cfg['model_dir'], "mean_neg_scores.dict"), "wb"))

            print("Finished saving utterance predictions")

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
