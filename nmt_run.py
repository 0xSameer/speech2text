# coding: utf-8

from basics import *
from nn_config import *
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


program_descrp = """
run nmt experiments
"""

'''
example:
python nmt_run.py -o $PWD/out -e 2 -k fisher_train

'''
xp = cuda.cupy if gpuid >= 0 else np

model = SpeechEncoderDecoder(gpuid)
# model_1 = SpeechEncoderDecoder(gpuid_2)

model.to_gpu(gpuid)
# model_1.to_gpu(gpuid_2)

if OPTIMIZER_ADAM1_SGD_0:
    print("using ADAM optimizer")
    # optimizer = optimizers.Adam(alpha=0.001,
    #                             beta1=0.9,
    #                             beta2=0.999,
    #                             eps=1e-08)
    # optimizer = optimizers.AdaGrad()
    optimizer = optimizers.Adam()
    optimizer.setup(model)
else:
    print("using SGD optimizer")
    optimizer = optimizers.SGD(lr=LEARNING_RATE)
    optimizer.setup(model)

if WEIGHT_DECAY:
    optimizer.add_hook(chainer.optimizer.WeightDecay(WD_RATIO))

# gradient clipping
optimizer.add_hook(chainer.optimizer.GradientClipping(threshold=10))
# optimizer.add_hook(chainer.optimizer.GradientNoise(eta=0.3))

def get_batch(m_dict, x_key, y_key,
              utt_list, vocab_dict,
              max_enc, max_dec, cat_speech_path=''):

    batch_data = {'X':[], 'y':[]}
    for u in utt_list:
        # for speech data
        x_data_found = False
        if x_key == 'sp':
            utt_sp_path = os.path.join(cat_speech_path,
                                      "{0:s}.npy".format(u))
            if not os.path.exists(utt_sp_path):
                utt_sp_path = os.path.join(cat_speech_path,
                                           u.split('_',1)[0],
                                           "{0:s}.npy".format(u))
            if os.path.exists(utt_sp_path):
                x_data_found = True
                x_data = xp.load(utt_sp_path)
                batch_data['X'].append(x_data[:max_enc])
            else:
                print("ERROR!! file not found: {0:s}".format(utt_sp_path))
        else:
            u_x_ids = [vocab_dict[x_key]['w2i'].get(w, UNK_ID) for w in m_dict[u][x_key]]
            u_x_ids = xp.asarray(u_x_ids, dtype=xp.int32)
            x_data_found = True
            batch_data['X'].append(u_x_ids[:max_enc])
        #  add english labels

        if x_data_found:
            if type(m_dict[u][y_key]) == list:
                en_ids = [vocab_dict[y_key]['w2i'].get(w, UNK_ID) for w in m_dict[u][y_key]]
            else:
                # dev and test data have multiple translations
                # choose the first one for computing perplexity
                en_ids = [vocab_dict[y_key]['w2i'].get(w, UNK_ID) for w in m_dict[u][y_key][0]]
            u_y_ids = [GO_ID] + en_ids[:MAX_EN_LEN-2] + [EOS_ID]
            if x_data_found == True:
                batch_data['y'].append(xp.asarray(u_y_ids, dtype=xp.int32))
    # end for all utterances in batch
    if len(batch_data['X']) > 0 and len(batch_data['y']) > 0:
        batch_data['X'] = F.pad_sequence(batch_data['X'], padding=PAD_ID)
        batch_data['y'] = F.pad_sequence(batch_data['y'], padding=PAD_ID)
    return batch_data


def feed_model(m_dict, b_dict, batch_size, vocab_dict,
               x_key, y_key, train, cat_speech_path, use_y=True,
               mini=False):
    # number of buckets
    num_b = b_dict['num_b']
    width_b = b_dict['width_b']

    if mini:
        # shuffle buckets
        if not SHUFFLE_BATCHES:
            random.seed(RANDOM_SEED_VALUE)
        # leave out the last bucket as it includes pruned utterances
        b_shuffled = list(range(num_b))
    else:
        b_shuffled = list(range(num_b))

    random.shuffle(b_shuffled)

    pred_sents = []
    utts = []
    total_loss = 0
    loss_per_epoch = 0
    total_loss_updates= 0

    sys.stderr.flush()

    total_utts = 0
    utt_list_batches = []
    for b in b_shuffled:
        if enc_key == 'sp':
            if b < num_b // 3:
                b_size = batch_size // BATCH_SIZE_SCALE
            elif (b >= num_b // 3) and (b < ((num_b*2) // 3)):
                if batch_size > BATCH_SIZE_MEDIUM:
                    b_size = BATCH_SIZE_MEDIUM // BATCH_SIZE_SCALE
                else:
                    b_size = batch_size // BATCH_SIZE_SCALE
            else:
                if batch_size > BATCH_SIZE_SMALL:
                    b_size = BATCH_SIZE_SMALL // BATCH_SIZE_SCALE
                else:
                    b_size = batch_size // BATCH_SIZE_SCALE
        else:
            if b < num_b // 2:
                b_size = batch_size
            elif (b >= num_b // 3) and (b < ((num_b*2) // 3)):
                b_size = batch_size
            else:
                b_size = batch_size

        bucket = b_dict['buckets'][b]
        if mini:
            # select % of the dataset for training
            bucket = random.sample(bucket, len(bucket) // TRAIN_SIZE_SCALE)

        b_len = len(bucket)
        total_utts += b_len
        random.shuffle(bucket)
        for i in range(0,b_len, b_size):
            utt_list_batches.append((bucket[i:i+b_size],b))
        # end bucket loop
    # end all buckets loop

    random.shuffle(utt_list_batches)

    with tqdm(total=total_utts, ncols=80) as pbar:
        for i, (utt_list, b) in enumerate(utt_list_batches):
            utt_list_0 = utt_list[:len(utt_list)//2]
            utt_list_1 = utt_list[len(utt_list)//2:]

            # get batch_data
            batch_data_0 = get_batch(m_dict,
                                   x_key, y_key,
                                   utt_list,
                                   vocab_dict,
                                   ((b+1) * width_b),
                                   (num_b * width_b),
                                   cat_speech_path=cat_speech_path)
            # batch_data_1 = get_batch(m_dict,
            #                        x_key, y_key,
            #                        utt_list_1,
            #                        vocab_dict,
            #                        ((b+1) * width_b),
            #                        (num_b * width_b),
            #                        cat_speech_path=cat_speech_path)

            # if (len(batch_data_0['X']) > 0 and len(batch_data_0['y']) > 0 and
            #                len(batch_data_1['X']) > 0 and len(batch_data_1['y']) > 0):
            if (len(batch_data_0['X']) > 0 and len(batch_data_0['y']) > 0):
                # batch_data_0['X'].to_gpu(gpuid)
                # batch_data_0['y'].to_gpu(gpuid)
                # batch_data_1['X'].to_gpu(gpuid_2)
                # batch_data_1['y'].to_gpu(gpuid_2)

                if use_y:
                    with chainer.using_config('train', train):
                        cuda.get_device(gpuid).use()
                        p0, loss_0 = model.forward(batch_data_0['X'], batch_data_0['y'])
                        loss_val = float(loss_0.data) / batch_data_0['y'].shape[1]

                        # cuda.get_device(gpuid_2).use()
                        # p1, loss_1 = model_1.forward(batch_data_1['X'], batch_data_1['y'])
                        # # store loss values for printing
                        # loss_val = (float(loss_0.data) + float(loss_1.data)) / (batch_data_0['y'].shape[1] + batch_data_1['y'].shape[1])
                else:
                    with chainer.using_config('train', False):
                        cuda.get_device(gpuid).use()
                        p0, _ = model.forward(batch_data_0['X'])
                        # cuda.get_device(gpuid_2).use()
                        # p1, _ = model_1.forward(batch_data_1['X'])
                        loss_val = 0.0

                utts.extend(utt_list)

                if len(p0) > 0:
                    pred_sents.extend(p0.tolist())

                # if len(p1) > 0:
                #     pred_sents.extend(p1.tolist())

                total_loss += loss_val
                total_loss_updates += 1

                loss_per_epoch = (total_loss / total_loss_updates)

                if train:
                    # set up for backprop
                    model.cleargrads()
                    # model_1.cleargrads()

                    loss_0.backward()
                    # loss_1.backward()

                    # model.addgrads(model_1)
                    optimizer.update()

                    # model_1.copyparams(model)


                out_str = "b={0:d},l={1:.2f},avg={2:.2f}".format((b+1),loss_val,loss_per_epoch)

                pbar.set_description('{0:s}'.format(out_str))
            else:
                print("no data in batch")
                print(len(batch_data_0['X']),
                      len(batch_data_0['y']),
                      len(batch_data_1['X']),
                      len(batch_data_1['y']))
                print(utt_list)

            pbar.update(len(utt_list))
        # end for batches
    # end tqdm
    return pred_sents, utts, loss_per_epoch
# end feed_model

def check_model():
    max_epoch = 0

    model_files = [f for f in os.listdir(os.path.dirname(model_fil))
                   if os.path.basename(model_fil).replace('.model','') in f]
    if len(model_files) > 0:
        print("-"*80)
        max_model_fil = max(model_files, key=lambda s: int(s.split('_')[-1].split('.')[0]))
        max_model_fil = os.path.join(os.path.dirname(model_fil),
                                     max_model_fil)
        print('model found = \n{0:s}'.format(max_model_fil))
        print('loading ...')
        serializers.load_npz(max_model_fil, model)
        # serializers.load_npz(max_model_fil, model_1)
        print("finished loading ..")
        max_epoch = int(max_model_fil.split('_')[-1].split('.')[0])
    else:
        print("-"*80)
        print('model not found')
    # end if model found

    return max_epoch
# end check_model

def train_loop(out_path, epochs, key, last_epoch, use_y, mini):
    if "fisher" in key:
        dev_key = "fisher_dev"
    else:
        dev_key = "callhome_devtest"

    with open(log_train_fil_name, mode='a') as train_log, open(log_dev_fil_name, mode='a') as dev_log:
        for i in range(epochs):
            print("-"*80)
            print("EPOCH = {0:d} / {1:d}".format(last_epoch+i+1, last_epoch+epochs))

            if (last_epoch+i+1 >= ITERS_TO_WEIGHT_NOISE) and (ITERS_TO_WEIGHT_NOISE > 0):
                print("Adding Gaussian weight noise, mean={0:.2f}, stdev={1:0.6f}".format(WEIGHT_NOISE_MU, WEIGHT_NOISE_SIGMA))
                model.add_weight_noise(WEIGHT_NOISE_MU, WEIGHT_NOISE_SIGMA)
                print("Finished adding Gaussian weight noise")
                # end adding gaussian weight noise

            # call train
            cat_speech_path = os.path.join(out_path, key)
            pred_sents, utts, train_loss = feed_model(map_dict[key],
                              b_dict=bucket_dict[key],
                              vocab_dict=vocab_dict,
                              batch_size=BATCH_SIZE,
                              x_key=enc_key,
                              y_key=dec_key,
                              train=True,
                              cat_speech_path=cat_speech_path,
                              use_y=use_y,
                              mini=mini)
            # log train loss
            train_log.write("{0:d}, {1:.4f}\n".format(last_epoch+i+1, train_loss))
            train_log.flush()
            os.fsync(train_log.fileno())

            # compute dev loss
            cat_speech_path = os.path.join(out_path, dev_key)
            pred_sents, utts, dev_loss = feed_model(map_dict[dev_key],
                              b_dict=bucket_dict[dev_key],
                              vocab_dict=vocab_dict,
                              batch_size=BATCH_SIZE,
                              x_key=enc_key,
                              y_key=dec_key,
                              train=False,
                              cat_speech_path=cat_speech_path,
                              use_y=use_y,
                              mini=False)

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

            # save model
            if ((i+1) % ITERS_TO_SAVE == 0) or (i == (epochs-1)):
                print("Saving model")
                serializers.save_npz(model_fil.replace(".model", "_{0:d}.model".format(last_epoch+i+1)), model)
                print("Finished saving model")
            # end if save model

            print("learning rate: {0:.6f}, optimizer: {1:s}, teacher_forcing_ratio: {2:.2f}".format(LEARNING_RATE,
                                "ADAM" if OPTIMIZER_ADAM1_SGD_0 else "SGD",
                                teacher_forcing_ratio))
            print("using GPU={0:d}".format(gpuid))
            print('model file name: {0:s}'.format(model_fil))
            print('dev log file name: {0:s}'.format(log_dev_fil_name))
            if ADD_NOISE:
                print('Adding gaussian noise to speech with stddev={0:.6f}'.format(NOISE_STDEV))

            print("-")
            print("-"*80)

        # end for epochs
    # end open log files
# end train loop


def my_main(out_path, epochs, key, use_y, mini):
    train = True if 'train' in key else False

    # check for existing model
    last_epoch = check_model()

    # check output file directory
    print(out_path)
    if not os.path.exists(out_path):
        print("{0:s} does not exist. Exiting".format(out_path))
        return 0
    # end if

    cat_speech_path = os.path.join(out_path, key)

    if (last_epoch >= ITERS_GRAD_NOISE) and ITERS_GRAD_NOISE > 0:
        print("------ Adding gradient noise")
        optimizer.add_hook(chainer.optimizer.GradientNoise(eta=GRAD_NOISE_ETA))
        print("Finished adding gradient noise")

    if train:
        train_loop(out_path, epochs, key, last_epoch, use_y=use_y, mini=mini)
    else:
        # call compute perplexity
        print("-"*80)
        print("EPOCH = {0:d}".format(last_epoch+1))
        pred_sents, utts, loss = feed_model(map_dict[key],
                          b_dict=bucket_dict[key],
                          vocab_dict=vocab_dict,
                          batch_size=BATCH_SIZE,
                          x_key=enc_key,
                          y_key=dec_key,
                          train=False,
                          cat_speech_path=cat_speech_path,
                          use_y=use_y,
                          mini=False)

        print("{0:s} {1:s} mean loss={2:.4f}".format("*" * 10,
                                            "train" if train else "dev",
                                            loss))
        print("-")
        print("-"*80)
        # print(model.cnn_bn.avg_mean)
        # print(model.cnn_bn.avg_var)

    print("all done ...")
# end my_main


def get_en_words_from_list(l):
    if STEMMIFY == False:
        return [w.decode() for w in l]
    else:
        return [stem(w.decode()) for w in l]

def calc_bleu(m_dict,
              v_dict,
              preds,
              utts,
              dec_key,
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
#     print("{0:10s} | {1:8.2f}".format("precision", *p))
#     print("{0:10s} | {1:8.2f}".format("recall", *r))


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


def test_func(out_path, batch_size, bucket_id, num_sent):
    m_dict = map_dict["fisher_train"]
    b_dict = bucket_dict["fisher_train"]

    cat_speech_path = os.path.join(out_path, "fisher_train")

    # number of buckets
    num_b = b_dict['num_b']
    width_b = b_dict['width_b']

    pred_sents = []
    utts = []
    total_loss = 0
    loss_per_epoch = 0
    total_loss_updates= 0

    sys.stderr.flush()
    total_utts = len(b_dict['buckets'][bucket_id])

    # a = input("pehla")
    # print("a")

    with tqdm(total=min(total_utts, num_sent)) as pbar:
        bucket = b_dict['buckets'][bucket_id]
        # random.shuffle(bucket)
        b_len = len(bucket)
        for i in range(0, min(b_len, num_sent), batch_size):
            utt_list = bucket[i:i+batch_size]
            utts.extend(utt_list)
            # get batch_data
            batch_data = get_batch(m_dict,
                                   enc_key, dec_key,
                                   utt_list,
                                   vocab_dict,
                                   ((bucket_id+1) * width_b),
                                   (num_b * width_b),
                                   cat_speech_path=cat_speech_path)

            # a = input("bhook")
            # print("a")

            with chainer.using_config('train', True):
                _, loss = model.forward(batch_data['X'], batch_data['y'])
            # store loss values for printing
            loss_val = float(loss.data)

            total_loss += loss_val
            total_loss_updates += 1

            loss_per_epoch = (total_loss / total_loss_updates)

            # a = input("doosra")
            # print("a")

            # set up for backprop
            model.cleargrads()
            loss.backward()
            # update parameters
            optimizer.update()

            # a = input("maar diya")
            # print("a")

            batch_data = get_batch(m_dict,
                                   enc_key, dec_key,
                                   utt_list,
                                   vocab_dict,
                                   ((bucket_id+1) * width_b),
                                   (num_b * width_b),
                                   cat_speech_path=cat_speech_path)
            if len(batch_data['X']) > 0 and len(batch_data['y']) > 0:
                with chainer.using_config('train', False):
                    p, _ = model.forward(batch_data['X'])

            # a = input("tijja")
            # print("a")

            if len(p) > 0:
                pred_sents.extend(p.tolist())

            # a = input("oooooooo")
            # print("a")

            out_str = "b={0:d},i={1:d}/{2:d},l={3:.2f},avg={4:.2f}".format((bucket_id+1),i,b_len,loss_val,loss_per_epoch)

            pbar.set_description('{0:s}'.format(out_str))

            pbar.update(len(utt_list))
        # end for batches
    # end tqdm

    return pred_sents, utts, loss_per_epoch
# end test_func

def main():
    parser = argparse.ArgumentParser(description=program_descrp)
    parser.add_argument('-o','--out_path', help='output path',
                        required=True)

    parser.add_argument('-e','--epochs', help='num epochs',
                        required=True)

    parser.add_argument('-k','--key', help='length/duration key cat, e.g. fisher_train, fisher_dev',
                        required=True)

    parser.add_argument('-y','--use_y', help='use y or just make predictions',
                        required=True)

    parser.add_argument('-m','--mini', help='use subset of training data',
                        required=True)

    args = vars(parser.parse_args())
    out_path = args['out_path']
    epochs = int(args['epochs'])
    key = args['key']
    use_y = False if int(args['use_y']) == 0 else True
    mini = False if int(args['mini']) == 0 else True

    print("number of epochs={0:d}".format(epochs))

    my_main(out_path, epochs, key, use_y, mini)
# end main

if __name__ == "__main__":
    main()
