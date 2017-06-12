
# coding: utf-8

# In[ ]:

from basics import *
from nn_config import *
from enc_dec import *

# In[ ]:

xp = cuda.cupy if gpuid >= 0 else np


# In[ ]:

text_data = pickle.load(open(text_data_dict, "rb"))


# In[ ]:

model = SpeechEncoderDecoder()

if gpuid >= 0:
    cuda.get_device(gpuid).use()
    model.to_gpu()

# In[ ]:

if OPTIMIZER_ADAM1_SGD_0:
    print("using ADAM optimizer")
    optimizer = optimizers.Adam(alpha=0.01,
                                beta1=0.9,
                                beta2=0.999,
                                eps=1e-08)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.5))
else:
    print("using SGD optimizer")
    optimizer = optimizers.SGD(lr=0.001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.01))

# gradient clipping
optimizer.add_hook(chainer.optimizer.GradientClipping(threshold=5))

print("loading data")
speech_feats = {}
speech_feats['train'] = xp.load(os.path.join(speech_dir, "train.npz"))
speech_feats['dev'] = xp.load(os.path.join(speech_dir, "dev.npz"))
speech_feats['test'] = xp.load(os.path.join(speech_dir, "test.npz"))
print("finished loading data")

# In[ ]:

log_train_fil_name, text_fname, dev_fname, test_fname

# In[ ]:

def create_speech_buckets():
    pass

# In[ ]:
def display_buckets(bucket_lengths, width_b = SPEECH_BUCKET_WIDTH):
    headings = ("ix", "len b", "num", "max fr", "avg fr", "max en", "avg en")
    print("{0:3s} | {1:5s} | {2:5s} | {3:6s} | {4:8s} | {5:6s} | {6:8s}".format(*headings))
    print("\n".join(["{0:3d} | {1:5d} | {2:5d} | {3:6d} | {4:8.0f} | {5:6d} | {6:8.0f}".format(i[0], (i[0]+1)*width_b, *i[1]) for i in list(bucket_lengths.items())]))


def prepare_data(width_b = SPEECH_BUCKET_WIDTH,
                 num_b = SPEECH_NUM_BUCKETS,
                 speech=True,
                 num_sent=NUM_TRAINING_SENTENCES,
                 filname_b=speech_bucket_data_fname,
                 cat="train", display=False):

    buckets = [{"X_fwd":[],
                 "X_rev":[],
                 "y": []}
                 for i in range(num_b)]

    print("split data into {0:d} buckets, each of width={1:d}".format(num_b,
                                                                      width_b))

    sys.stdout.flush()

    with tqdm(total=num_sent) as pbar:
        for i, sp_fil in enumerate(sorted(list(text_data[cat].keys()))[:num_sent]):

            fr_ids, en_ids, sp_feat = get_data_item(sp_fil, cat=cat)

            len_en = len(en_ids)
            len_fr = len(fr_ids)

            sp_feat_fwd = xp.pad(sp_feat,
                                ((width_b - len(sp_feat) % width_b,0), (0,0)),
                                mode='constant')
            sp_feat_rev = xp.pad(xp.flipud(sp_feat),
                                ((width_b - len(sp_feat) % width_b,0), (0,0)),
                                mode='constant')
            len_speech = len(sp_feat_fwd)
            target_text = [GO_ID] + en_ids[:MAX_EN_LEN-2] + [EOS_ID]
            target_text = xp.asarray(target_text, dtype=xp.int32)

            # check if speech features are valid
            if xp.isnan(xp.sum(sp_feat_fwd)) == True:
                print('''file={0:s} has a nan value,
                      fr len={1:d}, en len={2:d}'''.format(sp_fil,
                                                    len_fr, len_en))
            else:
                # add to buckets
                indx_b = min(num_b-1, (len_speech // width_b)-1)
                max_b_len = (indx_b+1) * width_b
                # print("max_b_len", max_b_len)
                # print("sp_feat_fwd", sp_feat_fwd[:max_b_len].shape)
                # print(buckets[indx_b]["X_fwd"])
                # buckets[indx_b]["X_fwd"].append(sp_feat_fwd[:max_b_len])
                if len(buckets[indx_b]["X_fwd"]) == 0:
                    buckets[indx_b]["X_fwd"] = xp.expand_dims(sp_feat_fwd[:max_b_len], axis=0)
                else:
                    buckets[indx_b]["X_fwd"] = xp.concatenate((buckets[indx_b]["X_fwd"], xp.expand_dims(sp_feat_rev[:max_b_len], axis=0)), axis=0)
                if len(buckets[indx_b]["X_rev"]) == 0:
                    buckets[indx_b]["X_rev"] = xp.expand_dims(sp_feat_rev[:max_b_len], axis=0)
                else:
                    buckets[indx_b]["X_rev"] = xp.concatenate((buckets[indx_b]["X_rev"], xp.expand_dims(sp_feat_rev[:max_b_len], axis=0)), axis=0)
                #buckets[indx_b]["X_rev"].append(sp_feat_rev[:max_b_len])
                buckets[indx_b]["y"].append(target_text)

            pbar.update(1)

    # pad all "y" data
    print("padding labels")
    for indx_b in range(len(buckets)):
        # buckets[indx_b]["X_fwd"] = xp.asarray(buckets[indx_b]["X_fwd"], dtype=xp.float32)
        # buckets[indx_b]["X_rev"] = xp.asarray(buckets[indx_b]["X_rev"], dtype=xp.float32)
        buckets[indx_b]["X_fwd"] = F.swapaxes(buckets[indx_b]["X_fwd"], 1,2)
        buckets[indx_b]["X_rev"] = F.swapaxes(buckets[indx_b]["X_rev"], 1,2)
        buckets[indx_b]["y"] = F.pad_sequence(buckets[indx_b]["y"], padding=PAD_ID)
        print('''{0:d} items in bucket={1:d}, each of length={2:d}, max en ids={3:d}'''.format(len(buckets[indx_b]["X_fwd"]),
                                                               indx_b+1,
                                                               len(buckets[indx_b]["X_fwd"][0]),
                                                               len(buckets[indx_b]["y"][0])))
    # # Saving bucket data
    # if filname_b:
    #     print("Saving bucket data")
    #     pickle.dump(buckets, open(filname_b, "wb"))
    return buckets

# In[ ]:

def display_prediction(fr_line, en_line, pred_words, prec=0, rec=0):
    print("{0:s}".format("-"*50))
    print("{0:s} | {1:80s}".format("Src", fr_line.strip()))
    print("{0:s} | {1:80s}".format("Ref", en_line.strip()))

    if not CHAR_LEVEL:
        print("{0:s} | {1:80s}".format("Hyp", " ".join(pred_words)))
    else:
        print("{0:s} | {1:80s}".format("Hyp", "".join(pred_words)))

    print("{0:s}".format("-"*50))

    print("{0:s} | {1:0.4f}".format("precision", prec))
    print("{0:s} | {1:0.4f}".format("recall", rec))

    # if plot_name and use_attn:
    #     plot_attention(alpha_arr, fr_words, pred_words, plot_name)

def predict_sentence(speech_feat, en_ids, p_filt=0, r_filt=0):
    # get prediction
    pred_ids, alpha_arr = model.encode_decode_predict(speech_feat)

    pred_words = [i2w["en"][w].decode() if w != EOS_ID else " _EOS" for w in pred_ids]

    prec = 0
    rec = 0
    filter_match = False

    matches = count_match(en_ids, pred_ids)
    if EOS_ID in pred_ids:
        pred_len = len(pred_ids)-1
    else:
        pred_len = len(pred_ids)

    # subtract 1 from length for EOS id
    prec = (matches/pred_len) if pred_len > 0 else 0
    rec = matches/len(en_ids)

    filter_match = (prec >= p_filt and rec >= r_filt)

    return pred_words, matches, len(pred_ids), len(en_ids), filter_match


def predict(s=0, num=1, cat="train", display=True, plot=False, p_filt=0, r_filt=0):
    print("English predictions, s={0:d}, num={1:d}:".format(s, num))

    metrics = {"cp":[], "tp":[], "t":[]}

    filter_count = 0

    for i, sp_fil in enumerate(sorted(list(text_data[cat].keys()))[s:s+num]):
        if plot:
            plot_name = os.path.join(model_dir, "{0:s}_plot.png".format(sp_fil))
        else:
            plot_name=None

        fr_ids, en_ids, speech_feat = get_data_item(sp_fil, cat=cat)

        # make prediction
        pred_words, cp, tp, t, f = predict_sentence(speech_feat, en_ids,
                                                    p_filt=p_filt, r_filt=r_filt)
        metrics["cp"].append(cp)
        metrics["tp"].append(tp)
        metrics["t"].append(t)
        filter_count += (1 if f else 0)

        if display:
            fr_line, en_line = get_text_lines(sp_fil, cat=cat)
            print("-"*80)
            print("prediction for: {0:s}".format(sp_fil))
            display_prediction(fr_line, en_line, pred_words, prec=0, rec=0)

    print("sentences matching filter = {0:d}".format(filter_count))
    return metrics

def count_match(list1, list2):
    # each list can have repeated elements. The count should account for this.
    count1 = Counter(list1)
    count2 = Counter(list2)
    count2_keys = count2.keys()-set([UNK_ID, EOS_ID])
    common_w = set(count1.keys()) & set(count2_keys)
    matches = sum([min(count1[w], count2[w]) for w in common_w])
    return matches


# In[ ]:

def compute_pplx(cat="dev", num_sent=NUM_MINI_DEV_SENTENCES, batches=False):
    loss = 0
    num_words = 0

    if not batches:
        with tqdm(total=num_sent) as pbar:
            for i, sp_fil in enumerate(sorted(list(text_data[cat].keys()))[:num_sent]):
                sys.stderr.flush()
                out_str = "loss={0:.6f}".format(0)
                pbar.set_description(out_str)
                fr_ids, en_ids, speech_feat = get_data_item(sp_fil, cat=cat)

                if len(speech_feat) >= MIN_SPEECH_LEN and len(en_ids) > 0:
                    # compute loss
                    curr_loss = float(model.encode_decode_train(speech_feat, en_ids, train=False).data)
                    loss += curr_loss
                    num_words += len(en_ids)

                    out_str = "loss={0:.6f}".format(curr_loss)
                    pbar.set_description(out_str)
                pbar.update(1)
        # end of pbar
        loss_per_word = curr_loss / num_words
    else:
        loss_per_word = batch_training(num_sent,
                                       DEV_BATCH_SIZE_LOOKUP,
                                       buckets_dict[cat],
                                       buckets_len_dict[cat],
                                       SPEECH_BUCKET_WIDTH,
                                       epoch=0,
                                       train=False,
                                       cat=cat)

    pplx = 2 ** loss_per_word

    print("{0:s}".format("-"*50))
    print("{0:s} | {1:0.6f}".format("dev perplexity", pplx))
    print("{0:s}".format("-"*50))

    return pplx


# In[ ]:

def bleu_stats(hypothesis, reference):
    yield len(hypothesis)
    yield len(reference)
    for n in range(1,5):
        s_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)+1-n)])
        r_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)+1-n)])
        yield max([sum((s_ngrams & r_ngrams).values()), 0])
        yield max([len(hypothesis)+1-n, 0])


# Compute BLEU from collected statistics obtained by call(s) to bleu_stats
def bleu(stats):
    if len(list(filter(lambda x: x==0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum([math.log(float(x)/y) for x,y in zip(stats[2::2],stats[3::2])]) / 4.
    return math.exp(min([0, 1-float(r)/c]) + log_bleu_prec)


def compute_bleu(cat="dev", num_sent=NUM_MINI_DEV_SENTENCES):
    list_of_references = []
    list_of_hypotheses = []
    with tqdm(total=num_sent) as pbar:
        for i, sp_fil in enumerate(sorted(list(text_data[cat].keys()))[:num_sent]):
            sys.stderr.flush()
            out_str = "predicting sentence={0:d}".format(i)
            pbar.update(1)

            fr_ids, en_ids, speech_feat = get_data_item(sp_fil, cat=cat)
            fr_line, en_line = get_text_lines(sp_fil, cat=cat)

            # add reference translation
            reference_words = en_line.strip().split()
            list_of_references.append(reference_words)

            if len(speech_feat) >= MIN_SPEECH_LEN and len(en_ids) > 0:
                pred_ids, _ = model.encode_decode_predict(speech_feat)
                pred_words = [i2w["en"][w].decode() if w != EOS_ID else "" for w in pred_ids]
                if CHAR_LEVEL:
                    pred_words = "".join(pred_words)
                    pred_words = pred_words.split()

            else:
                pred_words = []
            list_of_hypotheses.append(pred_words)

    stats = [0 for i in range(10)]
    for (r,h) in zip(list_of_references, list_of_hypotheses):
        stats = [sum(scores) for scores in zip(stats, bleu_stats(h,r))]
    print("BLEU: %0.2f" % (100 * bleu(stats)))

    return (100 * bleu(stats))


# In[ ]:

def get_text_lines(sp_fil, cat="train"):
    _, fr_line = get_ids(text_data[cat][sp_fil]["es"])
    _, en_line = get_ids(text_data[cat][sp_fil]["en"])

    return fr_line, en_line


# In[ ]:

def get_ids(align_list, char_level=CHAR_LEVEL):
    words = [a.word for a in align_list]
    text_line = " ".join(words)

    if not char_level:
        symbols = [w.encode() for w in words]
    else:
        symbols = [c.encode() for c in list(text_line.strip())]

    return symbols, text_line


# In[ ]:

def get_data_item(sp_fil, cat="train"):
    fr_sent, _ = get_ids(text_data[cat][sp_fil]["es"])
    en_sent, _ = get_ids(text_data[cat][sp_fil]["en"])

    fr_ids = [w2i["fr"].get(w, UNK_ID) for w in fr_sent]
    en_ids = [w2i["en"].get(w, UNK_ID) for w in en_sent]

    # speech_feat = xp.load(os.path.join(speech_dir, sp_fil+speech_extn)).astype(xp.float32)

    speech_feat = speech_feats[cat][sp_fil]
    return fr_ids, en_ids, speech_feat

# In[ ]:

def batch_training(num_training,
                   BATCH_SIZE_LOOKUP,
                   buckets,
                   width_b,
                   epoch=0,
                   train=True,
                   cat="train"):

    with tqdm(total=(num_training)) as pbar:
        sys.stderr.flush()

        total_trained = 0
        loss_per_epoch = 0
        total_loss = 0
        total_loss_updates = 0
        total_words = 0

        for buck_indx in range(len(buckets)):
            if total_trained >= num_training:
                break

            left_to_train = num_training - total_trained
            items_in_bucket = len(buckets[buck_indx]['X_fwd'])

            items_to_train_in_bucket = min(left_to_train, items_in_bucket)

            batch_size = BATCH_SIZE_LOOKUP[buck_indx]

            for i in range(0, items_to_train_in_bucket, batch_size):
                sp_files_in_batch = [t[0] for t in buckets[buck_indx][i:i+batch_size]]

                # get the next batch of data
                local_total_words = 0
                batch_data = []
                for sp_fil in sp_files_in_batch:
                    _, en_ids, speech_feat = get_data_item(sp_fil, cat=cat)
                    # print(speech_feat.shape, len(en_ids))
                    # if len(speech_feat) >= MIN_SPEECH_LEN:
                    batch_data.append((speech_feat[:pad_size_speech], en_ids[:pad_size_en]))
                    local_total_words += len(en_ids[:pad_size_en])

                # compute loss
                if len(batch_data) > 0:
                    loss = model.encode_decode_train_batch(batch_data, pad_size_speech, pad_size_en, train=train)

                    # store loss values for printing
                    loss_val = float(loss.data)

                    total_loss += loss_val

                    total_trained += len(batch_data)
                    total_loss_updates += 1

                    loss_per_epoch = (total_loss / total_loss_updates)
                    total_words += local_total_words / len(batch_data)

                    if train:
                        # set up for backprop
                        model.cleargrads()
                        loss.backward()
                        # update parameters
                        optimizer.update()

                    # print(sp_files_in_batch)
                    # print(loss.data, math.isnan(loss.data))
                    # print("L0 after", model.L0_enc.lateral.W.data[:2,:5])
                    # print(batch_data)

                    out_str = "epoch={0:d}, bucket={1:d}, i={2:d}, loss={3:.4f}, mean loss={4:.4f}".format((epoch+1), (buck_indx+1), i,loss_val, (loss_per_epoch))
                    pbar.set_description(out_str)
                pbar.update(len(batch_data))
            # end for current bucket
        # end for all buckets
    # end with pbar
    print("-"*80)
    print("mean loss={0:.4f}".format(loss_per_epoch))
    print("-"*80)
    return (loss_per_epoch / total_words)

# In[ ]:

def train_loop(num_training,
               num_epochs,
               log_mode="a",
               last_epoch_id=0,
               batches=False):
    # Set up log file for loss
    log_dev_fil = open(log_dev_fil_name, mode=log_mode)
    log_dev_csv = csv.writer(log_dev_fil, lineterminator="\n")
    bleu_score = 0

    # initialize perplexity on dev set
    # save model when new epoch value is lower than previous
    pplx = float("inf")

    sys.stderr.flush()

    if batches == True:
        print("creating buckets")
        buckets, bucket_lengths = populate_buckets(display=True)

    # start epochs
    for epoch in range(num_epochs):
        if not batches:
            single_instance_training(num_training, epoch)
        else:
            batch_training(num_training, BATCH_SIZE_LOOKUP , buckets, bucket_lengths, SPEECH_BUCKET_WIDTH, epoch,
                train=True, cat="train")
        # end with pbar

        print("finished training on {0:d} sentences".format(num_training))
        print("{0:s}".format("-"*50))
        print("computing perplexity")
        pplx_new = compute_pplx(cat="dev",
                                num_sent=NUM_MINI_DEV_SENTENCES,
                                batches=batches)

        if (epoch+1) % ITERS_TO_SAVE == 0:
            # bleu_score = compute_bleu(cat="dev",
            #                           num_sent=NUM_MINI_DEV_SENTENCES)
            print("Saving model")
            serializers.save_npz(model_fil.replace(".model", "_{0:d}.model".format(last_epoch_id+epoch+1)), model)
            print("Finished saving model")


        # log pplx and bleu score
        log_dev_csv.writerow([(last_epoch_id+epoch+1), pplx_new, bleu_score])
        log_dev_fil.flush()

        if pplx_new > pplx:
            print("perplexity went up during training, breaking out of loop")
            break
        pplx = pplx_new
        print(log_dev_fil_name)
        print(model_fil.replace(".model", "_{0:d}.model".format(epoch+1)))

    print("Simple predictions (╯°□°）╯︵ ┻━┻")
    print("training set predictions")
    _ = predict(s=0, num=2, cat="train", display=True, plot=False, p_filt=0, r_filt=0)
    print("Simple predictions (╯°□°）╯︵ ┻━┻")
    print("dev set predictions")
    _ = predict(s=0, num=2, cat="dev", display=True, plot=False, p_filt=0, r_filt=0)

    print("Final saving model")
    serializers.save_npz(model_fil, model)
    print("Finished saving model")

    # close log file
    log_dev_fil.close()

    print(log_dev_fil_name)
    print(model_fil)


# In[ ]:

# forward_states = model[model.lstm_enc[-1]].h
# backward_states = model[model.lstm_rev_enc[-1]].h


# In[ ]:

# model.enc_states = F.concat((forward_states, backward_states), axis=1)


# In[ ]:
def start_here(num_training=1000, num_epochs=1, batches=False):
    max_epoch_id = 0
    if os.path.exists(model_fil):
        # check last saved epoch model:
        for fname in [f for f in os.listdir(model_dir) if f.endswith("")]:
            if model_fil != os.path.join(model_dir, fname) and model_fil.replace(".model", "") in os.path.join(model_dir, fname):
                try:
                    epoch_id = int(fname.split("_")[-1].replace(".model", ""))
                    if epoch_id > max_epoch_id:
                        max_epoch_id = epoch_id
                except:
                    print("{0:s} not a valid model file".format(fname))
        print("last saved epoch model={0:d}".format(max_epoch_id))

        if load_existing_model:
            print("loading model ...")
            serializers.load_npz(model_fil, model)
            print("finished loading: {0:s}".format(model_fil))
        else:
            print("""model file already exists!!
                Delete before continuing, or enable load_existing flag""".format(model_fil))
            return
    else:
        print("model not found")

    train_loop(num_training=num_training,
               num_epochs=num_epochs,
               last_epoch_id=max_epoch_id,
               batches=batches)



def test_gradients(buckets):
    batch_size = 2
    for i in range(0, batch_size, batch_size):
        sp_files_in_batch = [t[0] for t in buckets[0][i:i+batch_size]]

        # print(pad_size_speech, pad_size_en, batch_size)
        # print("in bucket={0:d}, indx={1:d} to {2:d}".format(buck_indx, i, i+batch_size))

        # get the next batch of data
        batch_data = []
        for sp_fil in sp_files_in_batch:
            _, en_ids, speech_feat = get_data_item(sp_fil, cat="train")
            # print(speech_feat.shape, len(en_ids))
            batch_data.append((speech_feat[:50], en_ids[:20]))

        # compute loss
        loss = model.encode_decode_train_batch(batch_data, 50, 20, train=True)

        # store loss values for printing
        loss_val = float(loss.data)

        # set up for backprop
        model.cleargrads()
        loss.backward()
        # update parameters
        #optimizer.update()
        print(loss_val)

# In[ ]:
print("Starting experiment")
print(log_dev_fil_name)
print(model_fil)
print("num sentences={0:d} and num epochs={1:d}".format(NUM_MINI_TRAINING_SENTENCES, NUM_EPOCHS))



# buckets, bucket_lengths = populate_buckets(display=True)
buckets_dict = {}
# buckets_len_dict = {}

# buckets_dict['dev'], buckets_len_dict['dev'] = populate_buckets(
#                                             width_b=DEV_SPEECH_BUCKET_WIDTH,
#                                             num_b=DEV_SPEECH_NUM_BUCKETS,
#                                             speech=True,
#                                             num_sent=NUM_DEV_SENTENCES,
#                                             filname_b=None,
#                                             cat="dev", display=False)

buckets_dict['dev'] = prepare_data(width_b=DEV_SPEECH_BUCKET_WIDTH,
                                    num_b=DEV_SPEECH_NUM_BUCKETS,
                                    speech=True,
                                    num_sent=NUM_DEV_SENTENCES,
                                    filname_b=None,
                                    cat="dev", display=False)


# start_here(num_training=NUM_MINI_TRAINING_SENTENCES, num_epochs=NUM_EPOCHS, batches=True)

# batch_training(100, BATCH_SIZE_LOOKUP , buckets, bucket_lengths, SPEECH_BUCKET_WIDTH, 0)

# In[ ]:
