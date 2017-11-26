# coding: utf-8
from basics import *
from enc_dec import *
from prettytable import PrettyTable
import argparse
import textwrap
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
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


def calc_bleu(m_dict, v_dict, preds, utts, dec_key,
              weights=(0.25, 0.25, 0.25, 0.25),
              ref_index=-1):
    en_hyp = []
    en_ref = []
    ref_key = 'en_w' if 'en_' in dec_key else 'es_w'
    src_key = 'es_w'
    for u in tqdm(utts, ncols=80):
        if type(m_dict[u][ref_key]) == list:
            en_ref.append([get_en_words_from_list(m_dict[u][ref_key])])
        else:
            if ref_index == -1:
                en_r_list = []
                for r in m_dict[u][ref_key]:
                    en_r_list.append(get_en_words_from_list(r))
                en_ref.append(en_r_list)
            else:
                en_ref.append([get_en_words_from_list(m_dict[u][ref_key][ref_index])])

    join_str = ' ' if dec_key.endswith('_w') else ''

    total_matching_len = 0

    for u, p in zip(utts, preds):
        total_matching_len += 1
        t_str = join_str.join([v_dict['i2w'][i].decode() for i in p])
        t_str = t_str[:t_str.find('_EOS')]
        en_hyp.append(t_str.split())

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

def count_match(list1, list2):
    # each list can have repeated elements. The count should account for this.
    count1 = Counter(list1)
    count2 = Counter(list2)
    count2_keys = count2.keys()-set([UNK_ID, EOS_ID])
    common_w = set(count1.keys()) & set(count2_keys)
    matches = sum([min(count1[w], count2[w]) for w in common_w])
    return matches

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
              max_enc, max_dec, input_path=''):
    batch_data = {'X':[], 'y':[]}
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
                x_data = xp.load(utt_sp_path)
                # truncate max length
                batch_data['X'].append(x_data[:max_enc])
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
            x_ids = xp.asarray(x_ids, dtype=xp.int32)
            batch_data['X'].append(x_ids[:max_enc])
            # -----------------------------------------------------------------
        # ---------------------------------------------------------------------
        #  add labels
        # ---------------------------------------------------------------------
        if type(m_dict[u][y_key]) == list:
            en_ids = [vocab_dict[y_key]['w2i'].get(w, UNK_ID) for w in m_dict[u][y_key]]
        else:
            # dev and test data have multiple translations
            # choose the first one for computing perplexity
            en_ids = [vocab_dict[y_key]['w2i'].get(w, UNK_ID) for w in m_dict[u][y_key][0]]
        y_ids = [GO_ID] + en_ids[:max_dec-2] + [EOS_ID]
        batch_data['y'].append(xp.asarray(y_ids, dtype=xp.int32))
        # ---------------------------------------------------------------------
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
            b_size = int(batch_size['max'] // batch_size['scale'])
        elif b < (num_b*2) // 3:
            b_size = int(batch_size['med'] // batch_size['scale'])
        else:
            b_size = int(batch_size['min'] // batch_size['scale'])
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
               train, input_path, max_dec, t_cfg, use_y=True):
    # number of buckets
    num_b = b_dict['num_b']
    width_b = b_dict['width_b']
    pred_sents = []
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
                        cuda.get_device(gpuid).use()
                        p, _ = model.forward(X=batch_data['X'])
                        loss_val = 0.0
                # -------------------------------------------------------------
                # add list of utterances used
                # -------------------------------------------------------------
                utts.extend(utt_list)
                # -------------------------------------------------------------
                if len(p) > 0:
                    pred_sents.extend(p.tolist())

                total_loss += loss_val
                total_loss_updates += 1
                loss_per_epoch = (total_loss / total_loss_updates)
                # -------------------------------------------------------------
                # train mode logic
                # -------------------------------------------------------------
                if train:
                    model.cleargrads()
                    loss.backward()
                    optimizer.update()
                # -------------------------------------------------------------
                out_str = "b={0:d},l={1:.2f},avg={2:.2f}".format((b+1),loss_val,loss_per_epoch)
                pbar.set_description('{0:s}'.format(out_str))
            else:
                print("no data in batch")
                print(utt_list)
            # update progress bar
            pbar.update(len(utt_list))
        # end for batches
    # end tqdm
    return pred_sents, utts, loss_per_epoch
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
    return map_dict, vocab_dict, bucket_dict

def check_model(model_cfg, train_cfg):
    xp = cuda.cupy if train_cfg['gpuid'] >= 0 else np
    # -------------------------------------------------------------------------
    # initialize new model
    # -------------------------------------------------------------------------
    model = SpeechEncoderDecoder(model_cfg, train_cfg['gpuid'])
    model.to_gpu(train_cfg['gpuid'])
    # -------------------------------------------------------------------------
    # set up optimizer
    # -------------------------------------------------------------------------
    if train_cfg['optimizer'] == OPT_ADAM:
        print("using ADAM optimizer")
        optimizer = optimizers.Adam(alpha=train_cfg['lr'],
                                    beta1=0.9,
                                    beta2=0.999,
                                    eps=1e-08)
    else:
        print("using SGD optimizer")
        optimizer = optimizers.SGD(lr=train_cfg['lr'])

    # attach optimizer
    optimizer.setup(model)
    # -------------------------------------------------------------------------
    # optimizer settings
    # -------------------------------------------------------------------------
    if model_cfg['l2'] > 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(model_cfg['l2']))
    
    # gradient clipping
    optimizer.add_hook(chainer.optimizer.GradientClipping(threshold=model_cfg['grad_clip']))

    # gradient noise
    if train_cfg['grad_noise_eta'] > 0:
        print("------ Adding gradient noise")
        optimizer.add_hook(chainer.optimizer.GradientNoise(eta=train_cfg['grad_noise_eta']))
        print("Finished adding gradient noise")
    # -------------------------------------------------------------------------
    # check last saved model
    # -------------------------------------------------------------------------
    max_epoch = 0
    model_fil = model_cfg['model_fname']
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
    else:
        print("-"*80)
        print('model not found')
    # end if model found
    # -------------------------------------------------------------------------
    return max_epoch, model, optimizer
# end check_model

def train_loop(m_cfg, t_cfg, epochs, use_y):
    train_key = m_cfg['train_set']
    dev_key = m_cfg['dev_set']
    batch_size=t_cfg['batch_size']
    enc_key=m_cfg['enc_key']
    dec_key=m_cfg['dec_key']
    # -------------------------------------------------------------------------
    # check for existing model
    # -------------------------------------------------------------------------
    last_epoch, model, optimizer = check_model(m_cfg, t_cfg)
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
            # -----------------------------------------------------------------
            # Check to add Gaussian weight noise
            # -----------------------------------------------------------------
            if (last_epoch+i+1 >= t_cfg['iter_weight_noise']) and (t_cfg['iter_weight_noise'] > 0):
                print("Adding Gaussian weight noise, mean={0:.2f}, stdev={1:0.6f}".format(t_cfg['weight_noise_mean'], ))
                model.add_weight_noise(t_cfg['weight_noise_mean'], t_cfg['weight_noise_sigma'])
                print("Finished adding Gaussian weight noise")
            # end adding gaussian weight noise
            # -----------------------------------------------------------------
            # train
            # -----------------------------------------------------------------
            input_path = os.path.join(m_cfg['data_path'], 
                                      m_cfg['train_set'])
            pred_sents, utts, train_loss = feed_model(model,
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
                                              use_y=True)
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
            pred_sents, utts, dev_loss = feed_model(model,
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
                                              use_y=True)

            dev_b_score, chr_f_score, _, _ = calc_bleu(map_dict[dev_key],
                                                       vocab_dict[dec_key],
                                                       pred_sents, utts,
                                                       dec_key)

            # log dev loss
            dev_log.write("{0:d}, {1:.4f}, {2:.4f}, {3:.4f}\n".format(last_epoch+i+1, dev_loss, dev_b_score, chr_f_score))
            dev_log.flush()
            os.fsync(dev_log.fileno())

            print("^"*80)
            print("{0:s} train avg loss={1:.4f}, dev avg loss={2:.4f}, dev bleu={3:.4f}".format("*" * 10, train_loss, dev_loss, dev_b_score))
            print("^"*80)
            # -----------------------------------------------------------------
            # save model
            # -----------------------------------------------------------------
            if ((i+1) % t_cfg['iters_save_model'] == 0) or (i == (epochs-1)):
                print("Saving model")
                serializers.save_npz(model_fil.replace(".model", "_{0:d}.model".format(last_epoch+i+1)), model)
                print("Finished saving model")
            # end if save model
            # -----------------------------------------------------------------
        # end for epochs
    # end open log files
# end train loop

def train_main(model_cfg, train_cfg, epochs):
    # -------------------------------------------------------------------------
    # check model path
    # -------------------------------------------------------------------------
    print(model_cfg['data_path'])
    if not os.path.exists(model_cfg['data_path']):
        print("{0:s} does not exist. Exiting".format(model_cfg['data_path']))
        return 0
    # end if
    # -------------------------------------------------------------------------
    # call train loop
    # -------------------------------------------------------------------------
    train_loop(m_cfg=model_cfg, 
               t_cfg=train_cfg, 
               epochs=epochs, 
               use_y=True)
    # -------------------------------------------------------------------------
    print("all done ...")
# end train_main

# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=program_descrp)
    parser.add_argument('-m','--cfg_path', help='path for model config',
                        required=True)
    parser.add_argument('-e','--epochs', help='num epochs',
                        required=True)

    args = vars(parser.parse_args())

    cfg_path = args['cfg_path']

    with open(os.path.join(cfg_path, "model_cfg.json"), "r") as model_f:
        model_cfg = json.load(model_f)

    with open(os.path.join(cfg_path, "train_cfg.json"), "r") as train_f:
        train_cfg = json.load(train_f)

    epochs = int(args['epochs'])

    print("number of epochs={0:d}".format(epochs))

    train_main(model_cfg, train_cfg, epochs)
# end main
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
# -----------------------------------------------------------------------------
