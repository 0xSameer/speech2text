# coding: utf-8

from basics import *
from nn_config import *

class SpeechEncoderDecoder(Chain):

    def __init__(self, n_speech_dim, vsize_dec,
                 nlayers_enc, nlayers_dec,
                 n_units, gpuid, attn=False):
        '''
        n_speech_dim: dimensions for the speech features
        vsize:   vocabulary size
        nlayers: # layers
        attn:    if True, use attention
        '''
        # Store GPU id
        self.gpuid = gpuid
        if gpuid >= 0:
            # print("here")
            cuda.get_device(gpuid).use()

        super(SpeechEncoderDecoder, self).__init__()
        # create masking array for pad id
        #--------------------------------------------------------------------
        # add encoder layers
        #--------------------------------------------------------------------
        scale = 2
        # add LSTM layers
        self.lstm_enc = ["L{0:d}_enc".format(i) for i in range(nlayers_enc)]
        # first LSTM layer takes speech features
        self.add_link(self.lstm_enc[0], L.LSTM(n_speech_dim, n_units))
        # add remaining layers
        for lstm_name in self.lstm_enc[1:]:
            self.add_link(lstm_name, L.LSTM(n_units*scale, n_units))

        # reverse LSTM layer
        self.lstm_rev_enc = ["L{0:d}_rev_enc".format(i) for i in range(nlayers_enc)]
        self.add_link(self.lstm_rev_enc[0], L.LSTM(n_speech_dim, n_units))
        for lstm_name in self.lstm_rev_enc[1:]:
            self.add_link(lstm_name, L.LSTM(n_units*scale, n_units))

        #--------------------------------------------------------------------
        # add decoder layers
        #--------------------------------------------------------------------

        # add embedding layer
        self.add_link("embed_dec", L.EmbedID(vsize_dec, n_units))

        # add LSTM layers
        self.lstm_dec = ["L{0:d}_dec".format(i) for i in range(nlayers_dec)]
        self.add_link(self.lstm_dec[0], L.LSTM(n_units, 2*n_units))
        for lstm_name in self.lstm_dec[1:]:
            self.add_link(lstm_name, L.LSTM(2*n_units, 2*n_units))

        if attn > 0:
            # add context layer for attention
            self.add_link("context", L.Linear(4*n_units, 2*n_units))
        self.attn = attn

        # add output layer
        self.add_link("out", L.Linear(2*n_units, vsize_dec))

        self.n_units = n_units

        # create masking array for pad id
        self.mask_pad_id = xp.ones(vsize_dec, dtype=xp.float32)
        # make the class weight for pad id equal to 0
        # this way loss will not be computed for this predicted loss
        self.mask_pad_id[0] = 0

    def reset_state(self):
        # reset the state of LSTM layers
        for lstm_name in self.lstm_enc + self.lstm_rev_enc + self.lstm_dec:
            self[lstm_name].reset_state()
        self.loss = 0

    def set_decoder_state(self):
        xp = cuda.cupy if self.gpuid >= 0 else np
        # set the hidden and cell state of the first LSTM in the decoder
        # concatenate cell state of both enc LSTMs
        c_state = F.concat((self[self.lstm_enc[-1]].c, self[self.lstm_rev_enc[-1]].c))
        # concatenate hidden state of both enc LSTMs
        h_state = F.concat((self[self.lstm_enc[-1]].h, self[self.lstm_rev_enc[-1]].h))
        # h_state = F.split((self.enc_states), [:len(self.enc_states.data)])[0]
        self[self.lstm_dec[0]].set_state(c_state, h_state)

    '''
    Function to feed an input word through the embedding and lstm layers
        args:
        embed_layer: embeddings layer to use
        lstm_layer:  list of lstm layer names
    '''
    def feed_lstm(self, lstm_in, lstm_layer_list, train):
        # feed into first LSTM layer
        # hs = self[lstm_layer_list[0]](embed_id)
        hs = F.dropout(self[lstm_layer_list[0]](lstm_in), ratio=0.2, train=train)
        # feed into remaining LSTM layers
        for lstm_layer in lstm_layer_list[1:]:
            hs = F.dropout(self[lstm_layer](hs), ratio=0.2, train=train)


    def feed_pyramidal_lstm(self, feat_in, lstm_layer, scale, train=True):
        xp = cuda.cupy if self.gpuid >= 0 else np
        # create empty array to store the output
        # the output is scaled by the scale factor
        n_out_states = feat_in.shape[0] // scale
        out_dim = self[lstm_layer].state_size

        for i in range(0, n_out_states):
            lateral_states = self[lstm_layer](feat_in[(i*scale)])
            for j in range(1, scale):
                out = self[lstm_layer](feat_in[(i*scale)+j])
                lateral_states = F.concat((lateral_states, out), axis=1)
            # concatenate and append lateral states into out states
            if i > 0:
                out_states = F.concat((out_states, lateral_states), axis=0)
            else:
                out_states = lateral_states
        return F.expand_dims(out_states,1)

    def encode_speech_lstm(self, speech_feat, lstm_layer_list, train=True):
        # pad the speech feat and adjust dims

        # _TODO_ can optimize the loops to save memory. Using nested loops for every successive LSTM layer. Feed 8 units to L0 at a time.

        # initialize layer 0 input as speech features
        L_states = Variable(xp.expand_dims(speech_feat, 1), volatile=not train)
        # print("speech", L_states.shape)
        # initial scale values, for every layer 
        # except the final, scale down by 2
        scale = [2]*len(lstm_layer_list[:-1]) + [1]
        # feed LSTM layer
        for i, lstm_layer in enumerate(lstm_layer_list):
            # print(lstm_layer, "before", L_states.shape)
            L_states = self.feed_pyramidal_lstm(L_states, lstm_layer=lstm_layer, scale=scale[i], train=train)
            # print(lstm_layer, "out", L_states.shape)

        return L_states


    def encode(self, speech_in, lstm_layer_list, train):
        self.feed_lstm(speech_in, lstm_layer_list, train)

    def decode(self, word, train):
        embed_id = self.embed_dec(word)
        self.feed_lstm(embed_id, self.lstm_dec, train)

    #-----------------------------------------------------------------
    # For SGD - Batch size = 1
    #--------------------------------------------------------------------
    def encode_list(self, speech_feat, train=True):
        xp = cuda.cupy if self.gpuid >= 0 else np
        
        # forward LSTM
        self.L_FWD_STATES = self.encode_speech_lstm(speech_feat, self.lstm_enc, train)

        self.L_REV_STATES = self.encode_speech_lstm(xp.flip(speech_feat, axis=0), self.lstm_rev_enc, train)

        # reverse the states to align them with forward encoder
        self.L_REV_STATES = xp.flip(self.L_REV_STATES, axis=0)

        return_shape = self.L_FWD_STATES.shape

        self.enc_states = F.concat((self.L_FWD_STATES, self.L_REV_STATES), axis=1)

        self.enc_states = F.reshape(self.enc_states, shape=(return_shape[0], 2*return_shape[2]))

    def compute_context_vector(self, batches=True):
        xp = cuda.cupy if self.gpuid >= 0 else np

        batch_size, n_units = self[self.lstm_dec[-1]].h.shape
        # attention weights for the hidden states of each word in the input list

        if batches:
            # masking pad ids for attention
            weights = F.batch_matmul(self.enc_states, self[self.lstm_dec[-1]].h)
            weights = F.where(self.mask, weights, self.minf)

            alphas = F.softmax(weights)

            # compute context vector
            cv = F.reshape(F.batch_matmul(F.swapaxes(self.enc_states, 2, 1), alphas),
                                         shape=(batch_size, n_units))
        else:
            # without batches
            alphas = F.softmax(F.matmul(self[self.lstm_dec[-1]].h, self.enc_states, transb=True))
            # compute context vector
            if self.attn == SOFT_ATTN:
                cv = F.batch_matmul(self.enc_states, F.transpose(alphas))
                cv = F.transpose(F.sum(cv, axis=0))
            else:
                print("nothing to see here ...")

        return cv, alphas

    #--------------------------------------------------------------------
    # For SGD - Batch size = 1
    #--------------------------------------------------------------------
    def encode_decode_train(self, speech_feat, out_word_list, train=True):
        xp = cuda.cupy if self.gpuid >= 0 else np
        self.reset_state()
        # Add GO_ID, EOS_ID to decoder input
        decoder_word_list = [GO_ID] + out_word_list + [EOS_ID]
        # encode list of words/tokens
        self.encode_list(speech_feat[:MAX_SPEECH_LEN], train=train)
        # initialize decoder LSTM to final encoder state
        self.set_decoder_state()
        # decode and compute loss
        # convert list of tokens into chainer variable list
        var_dec = (Variable(xp.asarray(decoder_word_list, dtype=xp.int32).reshape((-1,1)),
                            volatile=not train))
        # Initialise first decoded word to GOID
        # pred_word = Variable(xp.asarray([GO_ID], dtype=np.int32), volatile=not train)

        # compute loss
        self.loss = 0
        # decode tokens
        for curr_word_var, next_word_var in zip(var_dec, var_dec[1:]):
            self.decode(curr_word_var, train=train)
            if self.attn:
                cv, _ = self.compute_context_vector(batches=False)
                cv_hdec = F.concat((cv, self[self.lstm_dec[-1]].h), axis=1)
                ht = F.tanh(self.context(cv_hdec))
                predicted_out = self.out(ht)
            else:
                predicted_out = self.out(self[self.lstm_dec[-1]].h)
            # compute loss
            self.loss += F.softmax_cross_entropy(predicted_out, next_word_var)
        report({"loss":self.loss},self)

        return self.loss

    #--------------------------------------------------------------------
    # For SGD - Batch size = 1
    #--------------------------------------------------------------------
    def decoder_predict(self, start_word, max_predict_len=MAX_PREDICT_LEN):
        xp = cuda.cupy if self.gpuid >= 0 else np
        alpha_arr = xp.empty((0,self.enc_states.shape[0]), dtype=xp.float32)

        # return list of predicted words
        predicted_sent = []
        # load start symbol
        prev_word = Variable(xp.asarray([start_word], dtype=xp.int32), volatile=True)
        pred_count = 0
        pred_word = None

        # start pred loop
        while pred_count < max_predict_len and pred_word != (EOS_ID) and pred_word != (PAD_ID):
            self.decode(prev_word, train=False)

            if self.attn:
                cv, alpha_list = self.compute_context_vector(batches=False)
                # concatenate hidden state
                cv_hdec = F.concat((cv, self[self.lstm_dec[-1]].h), axis=1)
                # add alphas row
                alpha_arr = xp.vstack((alpha_arr, alpha_list.data))

                ht = F.tanh(self.context(cv_hdec))
                prob = F.softmax(self.out(ht))
            else:
                prob = F.softmax(self.out(self[self.lstm_dec[-1]].h))

            if self.gpuid >= 0:
                prob = cuda.to_cpu(prob.data)[0].astype(np.float64)
            else:
                prob = prob.data[0].astype(np.float64)
            #prob /= np.sum(prob)
            #pred_word = np.random.choice(range(len(prob)), p=prob)
            pred_word = np.argmax(prob)
            predicted_sent.append(pred_word)
            prev_word = Variable(xp.asarray([pred_word], dtype=xp.int32), volatile=True)
            pred_count += 1
        return predicted_sent, alpha_arr

    #--------------------------------------------------------------------
    # For SGD - Batch size = 1
    #--------------------------------------------------------------------
    def encode_decode_predict(self, speech_feat, max_predict_len=MAX_PREDICT_LEN):
        xp = cuda.cupy if self.gpuid >= 0 else np
        self.reset_state()
        # encode list of words/tokens
        # in_word_list_no_padding = [w for w in in_word_list if w != PAD_ID]
        # enc_states = self.encode_list(in_word_list, train=False)
        self.encode_list(speech_feat, train=False)
        # initialize decoder LSTM to final encoder state
        self.set_decoder_state()
        # decode starting with GO_ID
        predicted_sent, alpha_arr = self.decoder_predict(GO_ID, max_predict_len)
        return predicted_sent, alpha_arr


    #--------------------------------------------------------------------
    # For batch size > 1
    #--------------------------------------------------------------------
    def pad_array(self, data, lim, at_start=True):
        xp = cuda.cupy if self.gpuid >= 0 else np
        r, c = data.shape
        if r >= lim:
            return data[:lim]

        rows_to_pad = lim-r
        zero_arr = xp.zeros((rows_to_pad, c), dtype=xp.float32)
        if at_start:
            ret_data = xp.concatenate((zero_arr, data), axis=0)
        else:
            ret_data = xp.concatenate((data, zero_arr), axis=0)
        return ret_data

    #--------------------------------------------------------------------
    # For batch size > 1
    #--------------------------------------------------------------------
    def pad_list(self, data, lim, at_start=True):
        xp = cuda.cupy if self.gpuid >= 0 else np
        if at_start:
            ret_data = [PAD_ID]*(lim - len(data)) + data
        else:
            ret_data = data + [PAD_ID]*(lim - len(data))
        return xp.asarray(ret_data, dtype=xp.int32)

    #--------------------------------------------------------------------
    # For batch size > 1
    #--------------------------------------------------------------------
    def batch_feed_pyramidal_lstm(self, feat_in, lstm_layer, scale, train=True):
        xp = cuda.cupy if self.gpuid >= 0 else np
        # create empty array to store the output
        # the output is scaled by the scale factor
        in_size, batch_size, in_dim = feat_in.shape
        n_out_states = in_size // scale
        # out_dim = self[lstm_layer].state_size
        # out_states = Variable(xp.zeros((n_out_states, batch_size, in_dim), dtype=xp.float32), volatile=not train)


        print(n_out_states, feat_in.shape, scale)

        for i in range(0, n_out_states):
            lateral_states = self[lstm_layer](feat_in[(i*scale)])
            print("lat", lateral_states.shape)
            for j in range(1, scale):
                out = self[lstm_layer](feat_in[(i*scale)+j])
                print("out", out.shape)
                lateral_states = F.concat((lateral_states, out), axis=1)
                print("lat concat", lateral_states.shape)
            # out_states[i] = lateral_states
            # concatenate and append lateral states into out states
            if i > 0:
                print(out_states.shape, lateral_states.shape)
                out_states = F.concat((out_states, F.expand_dims(lateral_states, 0)), axis=0)
            else:
                out_states = F.expand_dims(lateral_states, 0)
        return out_states

    def encode_speech_batch_lstm(self, speech_feat_batch, lstm_layer_list, train=True):
        # pad the speech feat and adjust dims

        # _TODO_ can optimize the loops to save memory. Using nested loops for every successive LSTM layer. Feed 8 units to L0 at a time.

        # initialize layer 0 input as speech features
        L_states = Variable(speech_feat_batch, volatile=not train)
        # print("speech", L_states.shape)
        # initial scale values, for every layer 
        # except the final, scale down by 2
        scale = [2]*len(lstm_layer_list[:-1]) + [1]
        # feed LSTM layer
        for i, lstm_layer in enumerate(lstm_layer_list):
            print(lstm_layer, "before", L_states.shape)
            L_states = self.batch_feed_pyramidal_lstm(L_states, lstm_layer=lstm_layer, scale=scale[i], train=train)
            print(lstm_layer, "out", L_states.shape)

        return L_states

    def encode_batch(self, fwd_encoder_batch, rev_encoder_batch, train=True):
        # convert list of tokens into chainer variable list
        self.encode_speech_batch_lstm(fwd_encoder_batch, self.lstm_enc, train)

        first_entry = True

        seq_len, batch_size, in_dim = fwd_encoder_batch.shape

        if self.attn:
            self.mask = xp.expand_dims(fwd_encoder_batch != 0, -1)
            self.minf = Variable(xp.full((batch_size, seq_len, 1), -1000.,
                                 dtype=xp.float32), volatile=not train)

        # for all sequences in the batch, feed the characters one by one
        self.L_FWD_STATES = self.encode_speech_batch_lstm(fwd_encoder_batch, self.lstm_enc, train)

        self.L_REV_STATES = self.encode_speech_batch_lstm(rev_encoder_batch, self.lstm_rev_enc, train)

        # reverse the states to align them with forward encoder
        self.L_REV_STATES = xp.flip(self.L_REV_STATES, axis=0)

        return_shape = self.L_FWD_STATES.shape

        self.enc_states = F.concat((self.L_FWD_STATES, self.L_REV_STATES), axis=2)

        print("L_FWD_STATES", self.L_FWD_STATES.shape)
        print("L_REV_STATES", self.L_REV_STATES.shape)
        print("enc_states", self.enc_states.shape)

        # self.enc_states = F.reshape(self.enc_states, shape=(return_shape[0], 2*return_shape[2]))


    #--------------------------------------------------------------------
    # For batch size > 1
    #--------------------------------------------------------------------
    def decode_batch(self, decoder_batch, train=True):
        xp = cuda.cupy if self.gpuid >= 0 else np
        # convert list of tokens into chainer variable list
        var_dec = (Variable(decoder_batch, volatile=(not train)))

        loss = 0

        # for all sequences in the batch, feed the characters one by one
        for curr_word, next_word in zip(var_dec, var_dec[1:]):
            # encode tokens
            self.decode(curr_word, train)

            if self.attn:
                cv, _ = self.compute_context_vector()
                cv_hdec = F.concat((cv, self[self.lstm_dec[-1]].h), axis=1)
                ht = F.tanh(self.context(cv_hdec))
                predicted_out = self.out(ht)
            else:
                predicted_out = self.out(self[self.lstm_dec[-1]].h)

            loss_arr = F.softmax_cross_entropy(predicted_out, next_word,
                                               class_weight=self.mask_pad_id)
            loss += loss_arr

        return loss

    #--------------------------------------------------------------------
    # For batch size > 1
    #--------------------------------------------------------------------
    def encode_decode_train_batch(self, batch_data, src_lim, tar_lim, train=True):
        xp = cuda.cupy if self.gpuid >= 0 else np
        self.reset_state()

        batch_size = len(batch_data)

        in_shape_r, in_shape_c = batch_data[0][0].shape
        # print(len(batch_data), in_shape_r, in_shape_c)

        fwd_encoder_batch = xp.full((batch_size, src_lim, in_shape_c), PAD_ID, dtype=xp.float32)
        rev_encoder_batch = xp.full((batch_size, src_lim, in_shape_c), PAD_ID, dtype=xp.float32)
        decoder_batch = xp.full((batch_size, tar_lim+2), PAD_ID, dtype=xp.int32)

        for i, (src, tar) in enumerate(batch_data):
            fwd_encoder_batch[i] = self.pad_array(src, src_lim)
            rev_encoder_batch[i] = self.pad_array(xp.flip(src, axis=0), src_lim)

            tar_data = [GO_ID] + tar + [EOS_ID]
            decoder_batch[i] = self.pad_list(tar_data, tar_lim+2, at_start=False)

        print(fwd_encoder_batch.shape, rev_encoder_batch.shape, decoder_batch.shape)

        # swap axes for batch and total rows
        fwd_encoder_batch = xp.swapaxes(fwd_encoder_batch, 0,1)
        rev_encoder_batch = xp.swapaxes(rev_encoder_batch, 0,1)
        decoder_batch = xp.swapaxes(decoder_batch, 0,1)

        print(fwd_encoder_batch.shape, rev_encoder_batch.shape, decoder_batch.shape)

        print("trying encode")

        self.encode_batch(fwd_encoder_batch, rev_encoder_batch, train)



        # hahahaha = self.encode_speech_batch_lstm(fwd_encoder_batch[0], self.lstm_enc, train)
        # print(hahahaha.shape)
        # L0_out = self[self.lstm_enc[0]](fwd_encoder_batch[0])
        # print("L0_out shape", L0_out.shape)

        # self.encode_speech_batch_lstm(fwd_encoder_batch, self.lstm_enc, train)
        # encode_speech_batch_lstm(w, self.lstm_enc, train)


        # encode list of words/tokens
        # self.encode_batch(fwd_encoder_batch, rev_encoder_batch, train=train)


        # initialize decoder LSTM to final encoder state
        self.set_decoder_state()
        # decode and compute loss
        self.loss = self.decode_batch(decoder_batch, train=train)

        return self.loss


# In[ ]:

