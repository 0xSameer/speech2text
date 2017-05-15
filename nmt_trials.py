
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

model = SpeechEncoderDecoder(SPEECH_DIM, vocab_size_en, num_layers_enc, num_layers_dec,
                               hidden_units, gpuid, attn=use_attn)
if gpuid >= 0:
    cuda.get_device(gpuid).use()
    model.to_gpu()


# In[ ]:

# optimizer = optimizers.Adam()
# optimizer = optimizers.Adam(alpha=0.0001, beta1=0.9, beta2=0.999, eps=1e-08)
optimizer = optimizers.SGD(lr=0.0001)
optimizer.setup(model)
# gradient clipping
optimizer.add_hook(chainer.optimizer.GradientClipping(threshold=5))
# optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))


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


def populate_buckets(width_b = SPEECH_BUCKET_WIDTH, 
                     num_b = SPEECH_NUM_BUCKETS, 
                     speech=True, 
                     num_sent=NUM_TRAINING_SENTENCES,
                     filname_b=speech_bucket_data_fname,
                     cat="train", display=False):
    
    buckets = [[] for i in range(num_b)]
    
    print("Splitting data into {0:d} buckets, each of width={1:d}".format(num_b, width_b))

    with tqdm(total=num_sent) as pbar:
        for i, sp_fil in enumerate(sorted(list(text_data[cat].keys()))[:num_sent]):

            fr_ids, en_ids, speech_feat = get_data_item(sp_fil, cat=cat)

            len_en = len(en_ids)
            len_fr = len(fr_ids)
            len_speech = len(speech_feat)

            # check if speech features are valid
            if xp.isnan(xp.sum(speech_feat)) == True:
                print("file={0:s} has a nan value, fr len={1:d}, en len={2:d}".format(sp_fil, len_fr, len_en))
            else:
                # add to buckets
                indx_b = min(num_b-1, len_speech // width_b)
                buckets[indx_b].append((sp_fil, len_speech, len_fr, len_en))

            pbar.update(1)

    bucket_lengths = {i:(len(l), 
                    max(l, key=lambda t:t[2])[2],
                    np.mean([i[2]for i in l]),
                    max(l, key=lambda t:t[3])[3],
                    np.mean([i[3]for i in l]))
                     for i, l in enumerate(buckets)}
    if display:
        display_buckets(bucket_lengths, width_b)

    # Saving bucket data
    print("Saving bucket data")
    pickle.dump(buckets, open(filname_b, "wb"))
    return buckets, bucket_lengths


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

def compute_pplx(cat="dev", num_sent=NUM_MINI_DEV_SENTENCES):
    loss = 0
    num_words = 0
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
    # end of for num_sent
    
    loss_per_word = loss / num_words
    pplx = 2 ** loss_per_word
    random_pplx = vocab_size_en

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
        symbols = [w.encode() for w in text_line.strip()]
    else:
        symbols = [c.encode() for c in list(text_line.strip())]
    
    return symbols, text_line


# In[ ]:

def get_data_item(sp_fil, cat="train"):    
    fr_sent, _ = get_ids(text_data[cat][sp_fil]["es"])
    en_sent, _ = get_ids(text_data[cat][sp_fil]["en"])

    fr_ids = [w2i["fr"].get(w, UNK_ID) for w in fr_sent]
    en_ids = [w2i["en"].get(w, UNK_ID) for w in en_sent]

    speech_feat = xp.load(os.path.join(speech_dir, sp_fil+speech_extn))
    return fr_ids, en_ids, speech_feat


# In[ ]:

# predict(s=0, num=1, cat="train", display=True, plot=False, p_filt=0, r_filt=0)


# In[ ]:

# print(b" ".join(get_ids(text_data["train"]["041.004"]["en"])[0]))


# In[ ]:
def single_instance_training(num_training, epoch):
    with tqdm(total=num_training) as pbar:
        sys.stderr.flush()
        for i, sp_fil in enumerate(sorted(list(text_data["train"].keys()))[:num_training], start=1):
            loss_per_epoch = 0
            out_str = "epoch={0:d}, loss={1:.4f}, mean loss={2:.4f}".format(epoch+1, 0, 0)
            pbar.set_description(out_str)
            
            # get the word/character ids
            fr_ids, en_ids, speech_feat = get_data_item(sp_fil, cat="train")

            it = (epoch * num_training) + i

            if len(speech_feat) >= MIN_SPEECH_LEN:
                # compute loss
                loss = model.encode_decode_train(speech_feat, en_ids, train=True)

                # set up for backprop
                model.cleargrads()
                loss.backward()
                # update parameters
                optimizer.update()
                # store loss value for display
                loss_val = float(loss.data)
                loss_per_epoch += loss_val

                out_str = "epoch={0:d}, loss={1:.4f}, mean loss={2:.4f}".format(epoch+1, loss_val, (loss_per_epoch / i))
                pbar.set_description(out_str)
            pbar.update(1)
        # end for num_training
    # end with pbar

# In[ ]:

def batch_training(num_training, 
                   batch_size, 
                   buckets, 
                   bucket_lengths, 
                   width_b,
                   epoch):
    num_batches = num_training // batch_size

    curr_batch = 0
    curr_bucket = 0
    prev_buckets_len = 0

    train_count = 0

    with tqdm(total=(num_training)) as pbar:
        sys.stderr.flush()

        total_trained = 0
        loss_per_epoch = 0

        for buck_indx in range(len(buckets)):
            if total_trained >= num_training:
                break
            left_to_train = num_training - total_trained
            items_in_bucket = bucket_lengths[buck_indx][0]
            pad_size_speech = (buck_indx+1) * width_b
            pad_size_en = bucket_lengths[buck_indx][3]

            items_to_train_in_bucket = min(left_to_train, items_in_bucket)

            for i in range(0, items_to_train_in_bucket, batch_size):
                sp_files_in_batch = [t[0] for t in buckets[buck_indx][i:i+batch_size]]

                # print("L0 before", model.L0_enc.lateral.W.data[:2,:5])

                # get the next batch of data
                batch_data = []
                for sp_fil in sp_files_in_batch:
                    _, en_ids, speech_feat = get_data_item(sp_fil, cat="train")
                    batch_data.append((speech_feat, en_ids))

                # compute loss
                loss = model.encode_decode_train_batch(batch_data, pad_size_speech, pad_size_en, train=True)

                # store loss values for printing
                loss_val = float(loss.data)
                loss_per_epoch += loss_val

                total_trained += len(batch_data)

                # set up for backprop
                model.cleargrads()
                loss.backward()
                # update parameters
                optimizer.update()
                # store loss value for display
                # print(sp_files_in_batch)
                # print(loss.data, math.isnan(loss.data))
                # print("L0 after", model.L0_enc.lateral.W.data[:2,:5])

                # print(batch_data)


                out_str = "epoch={0:d}, loss={1:.4f}, mean loss={2:.4f}".format(epoch+1, loss_val, (loss_per_epoch / total_trained))
                pbar.set_description(out_str)
                pbar.update(len(batch_data))
            # end for current bucket
        # end for all buckets
    # end with pbar

# In[ ]:

def train_loop(num_training, num_epochs, log_mode="a", last_epoch_id=0, batches=False):
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
        buckets, bucket_lengths = populate_buckets(display=False)

    # start epochs
    for epoch in range(num_epochs):
        if not batches:
            single_instance_training(num_training, epoch)
        else:
            batch_training(num_training, BATCH_SIZE , buckets, bucket_lengths, epoch)
        # end with pbar

        print("finished training on {0:d} sentences".format(num_training))
        print("{0:s}".format("-"*50))
        print("computing perplexity")
        pplx_new = compute_pplx(cat="dev", num_sent=NUM_MINI_DEV_SENTENCES)

        if pplx_new > pplx:
            print("perplexity went up during training, breaking out of loop")
            break
        
        pplx = pplx_new
        print(log_dev_fil_name)
        print(model_fil.replace(".model", "_{0:d}.model".format(epoch+1)))

        if (epoch+1) % ITERS_TO_SAVE == 0:
            bleu_score = compute_bleu(cat="dev", num_sent=NUM_MINI_DEV_SENTENCES)
            print("Saving model")
            serializers.save_npz(model_fil.replace(".model", "_{0:d}.model".format(last_epoch_id+epoch+1)), 
                                 model)
            print("Finished saving model")

        # log pplx and bleu score
        log_dev_csv.writerow([(last_epoch_id+epoch+1), pplx_new, bleu_score])
        log_dev_fil.flush()
    
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
def start_here(num_training=1000, num_epochs=1):
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
    train_loop(num_training=num_training, 
               num_epochs=num_epochs,
               last_epoch_id=max_epoch_id)


# In[ ]:
print("Starting experiment")
print("num sentences={0:d} and num epochs={1:d}".format(NUM_MINI_TRAINING_SENTENCES, NUM_EPOCHS))

# start_here(num_training=NUM_MINI_TRAINING_SENTENCES, num_epochs=NUM_EPOCHS)

# In[ ]:
