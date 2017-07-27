# coding: utf-8

from basics import *
from nn_config import *
from enc_dec import *
from prettytable import PrettyTable
import argparse
import textwrap

program_descrp = """
run nmt experiments
"""

'''
example:
python nmt_run.py -o $PWD/out -e 2 -k fisher_train

'''
xp = cuda.cupy if gpuid >= 0 else np

model = SpeechEncoderDecoder()

if gpuid >= 0:
    cuda.get_device(gpuid).use()
    model.to_gpu()

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
    optimizer = optimizers.SGD(lr=0.01)
    optimizer.setup(model)

if WEIGHT_DECAY:
    optimizer.add_hook(chainer.optimizer.WeightDecay(WD_RATIO))

# gradient clipping
optimizer.add_hook(chainer.optimizer.GradientClipping(threshold=2))


def get_batch(m_dict, x_key, y_key,
              utt_list, vocab_dict,
              max_enc, max_dec, cat_speech_path=''):

    batch_data = {'X':[], 'y':[]}
    for u in utt_list:
        # for speech data
        if x_key == 'sp':
            utt_sp_path = os.path.join(cat_speech_path,
                                      "{0:s}.npy".format(u))
            if not os.path.exists(utt_sp_path):
                utt_sp_path = os.path.join(cat_speech_path,
                                           u.split('_',1)[0],
                                           "{0:s}.npy".format(u))
            x_data = xp.load(utt_sp_path)
            batch_data['X'].append(x_data[:max_enc])
        else:
            u_x_ids = [vocab_dict[x_key]['w2i'].get(w, UNK_ID) for w in m_dict[u][x_key]]
            u_x_ids = xp.asarray(u_x_ids, dtype=xp.int32)
            batch_data['X'].append(u_x_ids[:max_dec])
        #  add english labels
        if type(m_dict[u][y_key]) == list:
            en_ids = [vocab_dict[y_key]['w2i'].get(w, UNK_ID) for w in m_dict[u][y_key]]
        else:
            # dev and test data have multiple translations
            # choose the first one for computing perplexity
            en_ids = [vocab_dict[y_key]['w2i'].get(w, UNK_ID) for w in m_dict[u][y_key][0]]
        u_y_ids = [GO_ID] + en_ids[:MAX_EN_LEN-2] + [EOS_ID]
        batch_data['y'].append(xp.asarray(u_y_ids, dtype=xp.int32))
    # end for all utterances in batch
    batch_data['X'] = F.pad_sequence(batch_data['X'], padding=PAD_ID)
    batch_data['y'] = F.pad_sequence(batch_data['y'], padding=PAD_ID)
    return batch_data


def feed_model(m_dict, b_dict, batch_size, vocab_dict,
               x_key, y_key, train, cat_speech_path, use_y=True):
    # number of buckets
    num_b = b_dict['num_b']
    width_b = b_dict['width_b']
    b_shuffled = [i for i in range(num_b)]
    # shuffle buckets
    random.shuffle(b_shuffled)

    pred_sents = []
    total_loss = 0
    loss_per_epoch = 0
    total_loss_updates= 0

    sys.stderr.flush()
    total_utts = len(m_dict)
    with tqdm(total=total_utts) as pbar:
        for b in b_shuffled:
            bucket = b_dict['buckets'][b]
            random.shuffle(bucket)
            b_len = len(bucket)
            for i in range(0,b_len, batch_size):
                utt_list = bucket[i:i+batch_size]
                # get batch_data
                batch_data = get_batch(m_dict,
                                       x_key, y_key,
                                       utt_list,
                                       vocab_dict,
                                       ((b+1) * width_b),
                                       (num_b * width_b),
                                       cat_speech_path=cat_speech_path)
                if use_y:
                    with chainer.using_config('train', train):
                        p, loss = model.forward(batch_data['X'], batch_data['y'])
                    # store loss values for printing
                    loss_val = float(loss.data)
                else:
                    with chainer.using_config('train', False):
                        p, loss = model.forward(batch_data['X'])
                    loss_val = 0.0

                if len(p) > 0:
                    pred_sents.extend(p.tolist())

                
                # loss_val = float(loss.data)

                total_loss += loss_val
                total_loss_updates += 1

                loss_per_epoch = (total_loss / total_loss_updates)

                if train:
                    # set up for backprop
                    model.cleargrads()
                    loss.backward()
                    # update parameters
                    optimizer.update()

                out_str = "b={0:d},i={1:d}/{2:d},l={3:.2f},avg={4:.2f}".format((b+1),i,b_len,loss_val,loss_per_epoch)

                pbar.set_description('{0:s}'.format(out_str))

                pbar.update(len(utt_list))
            # end for batches
    # end tqdm
    return pred_sents, loss_per_epoch
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
        print("finished loading ..")
        max_epoch = int(max_model_fil.split('_')[-1].split('.')[0])
    else:
        print("-"*80)
        print('model not found')
    # end if model found

    return max_epoch
# end check_model

def train_loop(cat_speech_path, epochs, key, last_epoch, use_y):
    with open(log_train_fil_name, mode='a') as train_log, open(log_dev_fil_name, mode='a') as dev_log:
        for i in range(epochs):
            print("-"*80)
            print("EPOCH = {0:d}".format(last_epoch+i+1))
            # call train
            pred_sents, train_loss = feed_model(map_dict['fisher_train'],
                              b_dict=bucket_dict['fisher_train'],
                              vocab_dict=vocab_dict,
                              batch_size=BATCH_SIZE,
                              x_key=enc_key,
                              y_key=dec_key,
                              train=True,
                              cat_speech_path=cat_speech_path, use_y=use_y)
            # log train loss
            train_log.write("{0:d}, {1:.4f}\n".format(last_epoch+i+1, train_loss))
            train_log.flush()
            os.fsync(train_log.fileno())

            # compute dev loss
            pred_sents, dev_loss = feed_model(map_dict['fisher_dev'],
                              b_dict=bucket_dict['fisher_dev'],
                              vocab_dict=vocab_dict,
                              batch_size=BATCH_SIZE,
                              x_key=enc_key,
                              y_key=dec_key,
                              train=False,
                              cat_speech_path=cat_speech_path, use_y=use_y)

            # log dev loss
            dev_log.write("{0:d}, {1:.4f}\n".format(last_epoch+i+1, dev_loss))
            dev_log.flush()
            os.fsync(dev_log.fileno())

            print("{0:s} train avg loss={1:.4f}, dev avg loss={2:.4f}".format("*" * 10, train_loss, dev_loss))
            print("-")
            print("-"*80)

            # save model
            if ((i+1) % ITERS_TO_SAVE == 0) or (i == (epochs-1)):
                print("Saving model")
                serializers.save_npz(model_fil.replace(".model", "_{0:d}.model".format(last_epoch+i+1)), model)
                print("Finished saving model")
            # end if save model
            print("MODEL NAME: {0:s}".format(os.path.basename(model_fil)))
        # end for epochs
    # end open log files
# end train loop


def my_main(out_path, epochs, key, use_y):
    train = True if 'train' in key else False

    # check for existing model
    last_epoch = check_model()

    # check output file directory
    if not os.path.exists(out_path):
        print("{0:s} does not exist. Exiting".format(out_path))
        return 0
    # end if

    cat_speech_path = os.path.join(out_path, key)

    if train:
        train_loop(cat_speech_path, epochs, key, last_epoch, use_y=use_y)
    else:
        # call compute perplexity
        print("-"*80)
        print("EPOCH = {0:d}".format(last_epoch+1))
        pred_sents, loss = feed_model(map_dict[key],
                          b_dict=bucket_dict[key],
                          vocab_dict=vocab_dict,
                          batch_size=BATCH_SIZE,
                          x_key=enc_key,
                          y_key=dec_key,
                          train=False,
                          cat_speech_path=cat_speech_path, use_y=use_y)

        print("{0:s} {1:s} mean loss={2:.4f}".format("*" * 10,
                                            "train" if train else "dev",
                                            loss))
        print("-")
        print("-"*80)

    print("all done ...")
# end my_main



def display_words(m_dict, v_dict, preds, utts, dec_key):
    es_ref = []
    en_ref = []
    for u in utts:
        es_ref.append(" ".join([w.decode() for w in m_dict[u]['es_w']]))
        en_ref.append(" ".join([w.decode() for w in m_dict[u]['en_w']]))

    en_pred = []
    join_str = ' ' if dec_key.endswith('_w') else ''

    for p in preds:
        en_pred.append(join_str.join([v_dict['i2w'][i].decode() for i in p if i != EOS_ID]))

    for u, es, en, p in zip(utts, es_ref, en_ref, en_pred):
        # for reference, 1st word is GO_ID, no need to display
        print("Utterance: {0:s}".format(u))
        display_pp = PrettyTable(["cat","sent"], hrules=True)
        display_pp.align = "l"
        display_pp.header = False
        display_pp.add_row(["es ref", textwrap.fill(es,50)])
        display_pp.add_row(["en ref", textwrap.fill(en,50)])
        display_pp.add_row(["en pred", textwrap.fill(p,50)])

        print(display_pp)

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
            with chainer.using_config('train', True):
                _, loss = model.forward(batch_data['X'], batch_data['y'])
            # store loss values for printing
            loss_val = float(loss.data)


            batch_data = get_batch(m_dict,
                                   enc_key, dec_key,
                                   utt_list,
                                   vocab_dict,
                                   ((bucket_id+1) * width_b),
                                   (num_b * width_b),
                                   cat_speech_path=cat_speech_path)
            with chainer.using_config('train', False):
                p, _ = model.forward(batch_data['X'])

            if len(p) > 0:
                pred_sents.extend(p.tolist())

            total_loss += loss_val
            total_loss_updates += 1

            loss_per_epoch = (total_loss / total_loss_updates)

            # set up for backprop
            model.cleargrads()
            loss.backward()
            # update parameters
            optimizer.update()

            out_str = "b={0:d},i={1:d}/{2:d},l={3:.2f},avg={4:.2f}".format((bucket_id+1),i,b_len,loss_val,loss_per_epoch)

            pbar.set_description('{0:s}'.format(out_str))

            pbar.update(len(utt_list))
        # end for batches
    # end tqdm

    return pred_sents, utts, loss_per_epoch
# end feed_model

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

    args = vars(parser.parse_args())
    out_path = args['out_path']
    epochs = int(args['epochs'])
    key = args['key']
    use_y = False if int(args['use_y']) == 0 else True

    print("number of epochs={0:d}".format(epochs))

    my_main(out_path, epochs, key, use_y)
# end main

if __name__ == "__main__":
    main()