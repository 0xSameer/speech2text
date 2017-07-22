# coding: utf-8

from basics import *
from nn_config import *
from enc_dec import *

import argparse

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
            if not os.path.exists():
                utt_sp_path = os.path.join(cat_speech_path, 
                                           utt_id.split('_',1)[0], 
                                           "{0:s}.npy".format(utt_id))
            batch_data['X'].append(xp.load(utt_sp_path))
        else:
            u_x_ids = [vocab_dict[x_key]['w2i'][w] for w in m_dict[u][x_key]]
            batch_data['X'].append(xp.asarray(u_x_ids, dtype=xp.int32))
        #  add english labels
        en_ids = [vocab_dict[y_key]['w2i'][w] for w in m_dict[u][y_key]]
        u_y_ids = [GO_ID] + en_ids[:MAX_EN_LEN-2] + [EOS_ID]
        batch_data['y'].append(xp.asarray(u_y_ids, dtype=xp.int32))
    # end for all utterances in batch
    batch_data['X'] = F.pad_sequence(batch_data['X'], padding=PAD_ID)
    # batch_data['X'] = F.expand_dims(batch_data['X'], axis=0)
    batch_data['y'] = F.pad_sequence(batch_data['y'], padding=PAD_ID)
    # batch_data['y'] = F.expand_dims(batch_data['y'], axis=0)
    return batch_data


def feed_model(m_dict, b_dict, batch_size, vocab_dict,
               x_key, y_key, train, cat_speech_path=''):
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
                                       ((i+1) * width_b), 
                                       (num_b * width_b), 
                                       cat_speech_path=out_path)
                with chainer.using_config('train', train):
                    p, loss = model.forward(batch_data['X'], batch_data['y'])
                if len(p) > 0:
                    pred_sents.extend(p.tolist())

                # store loss values for printing
                loss_val = float(loss.data)

                total_loss += loss_val
                total_loss_updates += 1

                loss_per_epoch = (total_loss / total_loss_updates)

                if train:
                    # set up for backprop
                    model.cleargrads()
                    loss.backward()
                    # update parameters
                    optimizer.update()

                out_str = "bucket={0:d}, i={1:d}/{2:d}, loss={3:.2f}, mean loss={3:.2f}".format((b+1),i,b_len,loss_val,loss_per_epoch)

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


def my_main(out_path, epochs, key):
    train = True if 'train' in key else False

    log_fname = log_train_fil_name if train else log_dev_fil_name

    # check output file directory
    if not os.path.exists(out_path):
        print("{0:s} does not exist. Exiting".format(out_path))
        return 0
    # end if

    # check for existing model
    last_epoch = check_model()

    with open(log_fname, mode='a') as log_fil:

        for i in range(epochs):
            print("-"*80)
            print("EPOCH = {0:d}".format(last_epoch+i+1))
            _, loss = feed_model(map_dict[key], 
                              b_dict=bucket_dict[key], 
                              vocab_dict=vocab_dict, 
                              batch_size=BATCH_SIZE, 
                              x_key=enc_key, 
                              y_key=dec_key, 
                              train=True, 
                              cat_speech_path=out_path)

            print("{0:s} {1:s} mean loss={2:.4f}".format("*" * 10,
                                                "train" if train else "dev",
                                                loss))
            print("-")
            print("-"*80)

            # log loss
            log_fil.write("{0:d}, {1:.4f}\n".format(last_epoch+i+1, loss))
            log_fil.flush()
            os.fsync(log_fil.fileno())

            if ((i+1) % ITERS_TO_SAVE == 0) or (i == (epochs-1)):
                print("Saving model")
                serializers.save_npz(model_fil.replace(".model", "_{0:d}.model".format(last_epoch+i+1)), model)
                print("Finished saving model")
        # end for epochs
    # end with open log files

    print("all done ...")
# end my_main

def main():
    parser = argparse.ArgumentParser(description=program_descrp)
    parser.add_argument('-o','--out_path', help='output path',
                        required=True)

    parser.add_argument('-e','--epochs', help='num epochs',
                        required=True)

    parser.add_argument('-k','--key', help='length/duration key cat',
                        required=True)

    args = vars(parser.parse_args())
    out_path = args['out_path']
    epochs = int(args['epochs'])
    key = args['key']

    print("number of epochs={0:d}".format(epochs))

    my_main(out_path, epochs, key)
# end main

if __name__ == "__main__":
    main()