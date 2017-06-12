# coding: utf-8

from basics import *
from nn_config import *


class SpeechEncoderDecoder(Chain):

    def __init__(self):
        '''
        n_speech_dim: dimensions for the speech features
        vsize:   vocabulary size
        nlayers: # layers
        attn:    if True, use attention
        '''
        self.init_params()

        if self.gpuid >= 0:
            cuda.get_device(self.gpuid).use()

        super(SpeechEncoderDecoder, self).__init__()

        self.init_model()


    def init_params(self):
        # Store GPU id
        self.gpuid = gpuid

        #--------------------------------------------------------------------
        # initialize model
        #--------------------------------------------------------------------
        if lstm1_or_gru0:
            self.RNN = L.LSTM
        else:
            self.RNN = L.GRU

        self.lstm1_or_gru0 = lstm1_or_gru0

        self.speech_dim = SPEECH_DIM
        self.n_units = hidden_units
        self.embed_units = embedding_units

        self.attn =  use_attn

        self.vocab_size_en = vocab_size_en

        self.max_pool_stride = max_pool_stride
        self.max_pool_pad = max_pool_pad


    def add_rnn_layers(self, layer_names, in_units, out_units, scale):
        # add first layer
        self.add_link(layer_names[0], self.RNN(in_units, out_units))
        # add remaining layers
        for rnn_name in layer_names[1:]:
            self.add_link(rnn_name, self.RNN(out_units*scale, out_units))


    def add_cnn_layers(self, layer_name, in_channels, out_channels,
                       num_filters, ksize, stride, pad):
        self.add_link(layer_name,
                      L.ConvolutionND(ndim=1,
                                      in_channels=in_channels,
                                      out_channels=out_channels,
                                      ksize=2,
                                      stride=1,
                                      pad=1))


    def init_rnn_model(self, scale, in_dim):
        #--------------------------------------------------------------------
        # add encoder layers
        #--------------------------------------------------------------------
        self.rnn_enc = ["L{0:d}_enc".format(i) for i in range(num_layers_enc)]
        self.add_rnn_layers(self.rnn_enc,
                            in_dim,
                            self.n_units,
                            scale=scale)

        # reverse LSTM layer
        self.rnn_rev_enc = ["L{0:d}_rev_enc".format(i) for i in range(num_layers_enc)]
        self.add_rnn_layers(self.rnn_rev_enc,
                            in_dim,
                            self.n_units,
                            scale=scale)

        # add LSTM layers
        self.rnn_dec = ["L{0:d}_dec".format(i) for i in range(num_layers_dec)]
        self.add_rnn_layers(self.rnn_dec,
                            self.embed_units,
                            2*self.n_units,
                            scale=scale)

    def init_cnn_model(self):
        self.cnns = []
        # add CNN layers
        cnn_out_dim = 0
        for l in cnn_filters:
            lname = "CNN_{0:d}".format(l['ksize'])
            cnn_out_dim += l["out_channels"]
            self.cnns.append(lname)
            self.add_link(lname, L.ConvolutionND(**l))

        self.cnn_out_dim = cnn_out_dim

        # add highway layers
        self.highway = ["highway_{0:d}".format(i)
                         for i in range(num_highway_layers)]

        for hname in self.highway:
            self.add_link(hname, L.Highway(self.cnn_out_dim))


    def init_model(self):
        if MODEL_TYPE == MODEL_RNN:
            self.scale = 2
            rnn_in_units = self.speech_dim
        elif MODEL_TYPE == MODEL_CNN:
            self.scale = 1
            self.init_cnn_model()
            rnn_in_units = self.cnn_out_dim
        else:
            print("Nooooooooooooooooooo")

        # initialize RNN layers
        print("rnn_in_units", rnn_in_units)
        self.init_rnn_model(self.scale, rnn_in_units)

        # add embedding layer
        self.add_link("embed_dec", L.EmbedID(self.vocab_size_en,
                                             self.embed_units))

        if self.attn > 0:
            # add context layer for attention
            self.add_link("context", L.Linear(4*self.n_units, 2*self.n_units))

        # add output layer
        self.add_link("out", L.Linear(2*self.n_units, vocab_size_en))

        # create masking array for pad id
        self.mask_pad_id = xp.ones(self.vocab_size_en, dtype=xp.float32)
        # make the class weight for pad id equal to 0
        # this way loss will not be computed for this predicted loss
        self.mask_pad_id[0] = 0


    def reset_state(self):
        # reset the state of LSTM layers
        for rnn_name in self.rnn_enc + self.rnn_rev_enc + self.rnn_dec:
            self[rnn_name].reset_state()
        self.loss = 0

    def set_decoder_state(self):
        # set the hidden and cell state (if LSTM) of the first RNN in the decoder
        if self.lstm1_or_gru0:
            # concatenate cell state of both enc LSTMs
            c_state = F.concat((self[self.rnn_enc[-1]].c, self[self.rnn_rev_enc[-1]].c))
        # concatenate hidden state of both enc LSTMs
        h_state = F.concat((self[self.rnn_enc[-1]].h, self[self.rnn_rev_enc[-1]].h))
        if self.lstm1_or_gru0:
            self[self.rnn_dec[0]].set_state(c_state, h_state)
        else:
            self[self.rnn_dec[0]].set_state(h_state)


    def compute_context_vector(self, batches=True):
        batch_size, n_units = self[self.rnn_dec[-1]].h.shape
        # attention weights for the hidden states of each word in the input list

        if batches:
            weights = F.batch_matmul(self.enc_states, self[self.rnn_dec[-1]].h)
            # weights = F.where(self.mask, weights, self.minf)
            alphas = F.softmax(weights)
            # compute context vector
            cv = F.squeeze(F.batch_matmul(F.swapaxes(self.enc_states, 2, 1), alphas), axis=2)

        else:
            # without batches
            alphas = F.softmax(F.matmul(self[self.rnn_dec[-1]].h, self.enc_states, transb=True))
            # compute context vector
            if self.attn == SOFT_ATTN:
                cv = F.batch_matmul(self.enc_states, F.transpose(alphas))
                cv = F.transpose(F.sum(cv, axis=0))
            else:
                print("nothing to see here ...")

        return cv, alphas

    def pad_array(self, data, lim, at_start=True):
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

    def pad_list(self, data, lim, at_start=True):
        if len(data) >= lim:
            ret_data = data[:lim]
        else:
            rows_to_pad = lim-len(data)
            if at_start:
                ret_data = [PAD_ID]*(rows_to_pad) + data
            else:
                ret_data = data + [PAD_ID]*(rows_to_pad)
        return xp.asarray(ret_data, dtype=xp.int32)


    def feed_rnn(self, rnn_in, rnn_layers):
        # feed into first rnn layer
        hs = F.dropout(self[rnn_layers[0]](rnn_in), ratio=0.2)
        # feed into remaining rnn layers
        for rnn_layer in rnn_layers[1:]:
            hs = F.dropout(self[rnn_layer](hs), ratio=0.2)
        return hs


    def encode(self, data_in, rnn_layers):
        h = self.feed_rnn(data_in, rnn_layers)
        return h

    def decode(self, word):
        embed_id = self.embed_dec(word)
        h = self.feed_rnn(embed_id, self.rnn_dec)
        return h


    def batch_feed_pyramidal_rnn(self, feat_in, rnn_layer, scale):
        # create empty array to store the output
        # the output is scaled by the scale factor
        in_size, batch_size, in_dim = feat_in.shape
        n_out_states = in_size // scale

        # print(n_out_states, feat_in.shape, scale)

        for i in range(0, n_out_states):
            lateral_states = self[rnn_layer](feat_in[(i*scale)])
            for j in range(1, scale):
                out = self[rnn_layer](feat_in[(i*scale)+j])
                lateral_states = F.concat((lateral_states, out), axis=1)
            # concatenate and append lateral states into out states
            if i > 0:
                out_states = F.concat((out_states, F.expand_dims(lateral_states, 0)), axis=0)
            else:
                out_states = F.expand_dims(lateral_states, 0)
        return out_states

    def encode_speech_batch_lstm_seg(self, L_states, rnn_layer_list):
        # pad the speech feat and adjust dims
        # initialize layer 0 input as speech features
        # initial scale values, for every layer
        # except the final, scale down by 2
        scale = [self.scale]*len(rnn_layer_list[:-1]) + [1]
        # feed LSTM layer
        for i, rnn_layer in enumerate(rnn_layer_list):
            L_states = self.batch_feed_pyramidal_rnn(L_states, rnn_layer=rnn_layer, scale=scale[i])
        return L_states

    def encode_speech_batch_lstm(self, speech_feat_batch, rnn_layer_list):
        in_size, batch_size, in_dim = speech_feat_batch.shape
        # Optimize the loops to save memory. Using nested loops for every successive LSTM layer. Feed 8 units to L0 at a time.
        # step size to process input
        step_size = self.scale**len(rnn_layer_list[:-1])
        # print("step size", step_size)
        for s in range(0,in_size,step_size):
            if s == 0:
                out_states = self.encode_speech_batch_lstm_seg(speech_feat_batch[s:s+step_size], rnn_layer_list)
            else:
                out_states = F.concat((out_states, self.encode_speech_batch_lstm_seg(speech_feat_batch[s:s+step_size], rnn_layer_list)), axis=0)
        # end for step_size
        return out_states


    def encode_batch(self, fwd_encoder_batch, rev_encoder_batch=None):
        # convert list of tokens into chainer variable list
        seq_len, batch_size, in_dim = fwd_encoder_batch.shape
        print("fwd_encoder_batch", fwd_encoder_batch.shape)

        if self.attn:
            #self.mask = F.expand_dims(fwd_encoder_batch != 0, -1)
            self.minf = Variable(xp.full((batch_size, seq_len, 1), -1000.,
                                 dtype=xp.float32))

        # for all sequences in the batch, feed the characters one by one
        L_FWD_STATES = self.encode_speech_batch_lstm(fwd_encoder_batch,
                                                     self.rnn_enc)

        if not rev_encoder_batch:
            rev_encoder_batch = F.flipud(fwd_encoder_batch)
            print("rev_encoder_batch", rev_encoder_batch.shape)

        L_REV_STATES = self.encode_speech_batch_lstm(rev_encoder_batch,
                                                     self.rnn_rev_enc)

        # reverse the states to align them with forward encoder
        L_REV_STATES = F.flipud(L_REV_STATES)

        self.enc_states = F.concat((L_FWD_STATES, L_REV_STATES), axis=2)
        self.enc_states = F.swapaxes(self.enc_states, 0, 1)

    def decode_batch(self, decoder_batch):
        loss = 0
        # for all sequences in the batch, feed the characters one by one
        for curr_word, next_word in zip(decoder_batch, decoder_batch[1:]):
            # encode tokens
            self.decode(curr_word)

            if self.attn:
                cv, _ = self.compute_context_vector(batches=True)
                cv_hdec = F.concat((cv, self[self.rnn_dec[-1]].h), axis=1)
                ht = F.tanh(self.context(cv_hdec))
                predicted_out = self.out(ht)
            else:
                predicted_out = self.out(self[self.rnn_dec[-1]].h)

            loss_arr = F.softmax_cross_entropy(predicted_out, next_word,
                                               class_weight=self.mask_pad_id)
            loss += loss_arr

        return loss

    def prepare_batch(self, batch_data, src_lim, tar_lim):
        # pad and return batch data
        pass


    def encode_decode_train_batch(self, batch_data, src_lim, tar_lim):
        self.reset_state()

        batch_size = len(batch_data)

        # get the dimension of the input data
        # format of batch_data = num in batch * sequence length * frame/char/word dimensionality
        # batch_data is a list of (speech_feat data, en_ids and pad_size_en)
        # batch_data[0][0] gets the first speech_feat numpy array
        # the columns (dimensionality) of all speech feats in a batch
        # is assumed to be the same
        in_shape_r, in_shape_c = batch_data[0][0].shape

        fwd_encoder_batch = xp.full((batch_size, src_lim, in_shape_c), PAD_ID, dtype=xp.float32)
        rev_encoder_batch = xp.full((batch_size, src_lim, in_shape_c), PAD_ID, dtype=xp.float32)
        decoder_batch = xp.full((batch_size, tar_lim+2), PAD_ID, dtype=xp.int32)

        for i, (src, tar) in enumerate(batch_data):
            fwd_encoder_batch[i] = self.pad_array(src, src_lim)
            rev_encoder_batch[i] = self.pad_array(xp.flip(src, axis=0), src_lim)

            tar_data = [GO_ID] + tar + [EOS_ID]
            decoder_batch[i] = self.pad_list(tar_data, tar_lim+2, at_start=False)

        # free memory
        del batch_data

        # swap axes for batch and total rows
        fwd_encoder_batch = xp.swapaxes(fwd_encoder_batch, 0,1)
        rev_encoder_batch = xp.swapaxes(rev_encoder_batch, 0,1)
        decoder_batch = xp.swapaxes(decoder_batch, 0,1)

        # print(fwd_encoder_batch.shape, rev_encoder_batch.shape, decoder_batch.shape)

        self.encode_batch(fwd_encoder_batch, rev_encoder_batch)

        # initialize decoder LSTM to final encoder state
        self.set_decoder_state()
        # decode and compute loss
        self.loss = self.decode_batch(decoder_batch)

        self.enc_states = 0

        return self.loss


    def forward_cnn(self, X):
        # perform convolutions
        h = F.relu(self[self.cnns[0]](X))
        print(h.shape)
        for i in range(len(self.cnns[1:])):
            h = F.concat((h, F.relu(self[self.cnns[i]](X))), axis=1)
            print(h.shape)

        # max pooling
        h = F.max_pooling_nd(h, ksize=max_pool_stride,
                             stride=max_pool_stride,
                             pad=max_pool_stride//2)
        print(h.shape)

        # out dimension:
        # batch size * cnn out dim * num time frames after pooling
        return h


    def forward_highway(self, X):
        # highway
        for i in range(len(self.highway)):
            h = self[self.highway[i]](X)
            print(h.shape)
        return h

    def forward_rnn(self, X):
        self.reset_state()
        in_size, batch_size, in_dim = X.shape
        print("X", X.shape)
        for i in range(in_size):
            if i > 0:
                h_fwd = F.concat((h_fwd,
                                  F.expand_dims(self.encode(X[i],
                                    self.rnn_enc), 0)),
                                  axis=0)
                h_rev = F.concat((h_rev,
                                  F.expand_dims(self.encode(X[-i],
                                    self.rnn_rev_enc), 0)),
                                  axis=0)
            else:
                h_fwd = F.expand_dims(self.encode(X[i], self.rnn_enc), 0)
                h_rev = F.expand_dims(self.encode(X[-i], self.rnn_rev_enc), 0)

        h_rev = F.flipud(h_rev)
        self.enc_states = F.concat((h_fwd, h_rev), axis=2)
        self.enc_states = F.swapaxes(self.enc_states, 0, 1)
        return h_fwd, h_rev


    def forward(self, X):
        if MODEL_TYPE == MODEL_CNN:
            h = self.forward_cnn(X)
            print(h.shape)
            h = F.rollaxis(h, 2)
            print(h.shape)
        # h = self.forward_highway(h)
        self.forward_rnn(h)


# In[ ]:

