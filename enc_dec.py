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


    def feed_rnn(self, rnn_in, rnn_layers, highway_layers=None):
        # feed into first rnn layer
        # hs = F.dropout(self[rnn_layers[0]](rnn_in), ratio=0.2)
        hs = self[rnn_layers[0]](rnn_in)
        # feed into remaining rnn layers
        for rnn_layer in rnn_layers[1:]:
            # hs = F.dropout(self[rnn_layer](hs), ratio=0.2)
            hs = self[rnn_layer](hs)
        return hs

    def encode(self, data_in, rnn_layers):
        h = self.forward_highway(data_in)
        h = self.feed_rnn(h, rnn_layers)
        return h

    def decode(self, word):
        embed_id = self.embed_dec(word)
        h = self.feed_rnn(embed_id, self.rnn_dec)
        if self.attn:
            cv, _ = self.compute_context_vector(batches=True)
            cv_hdec = F.concat((cv, self[self.rnn_dec[-1]].h), axis=1)
            # ht = F.tanh(self.context(cv_hdec))
            ht = self.context(cv_hdec)
            predicted_out = self.out(ht)
        else:
            predicted_out = self.out(self[self.rnn_dec[-1]].h)
        return predicted_out

    def decode_batch(self, decoder_batch):
        loss = 0
        # for all sequences in the batch, feed the characters one by one
        for curr_word, next_word in zip(decoder_batch, decoder_batch[1:]):
            # encode tokens
            predicted_out = self.decode(curr_word)

            loss_arr = F.softmax_cross_entropy(predicted_out, next_word,
                                               class_weight=self.mask_pad_id)
            loss += loss_arr

        return loss

    def predict_batch(self, batch_size, pred_limit, y=None):
        # max number of predictions to make
        # if labels are provided, this variable is not used
        stop_limit = pred_limit
        # to track number of predictions made
        npred = 0
        # to store loss
        loss = 0
        # if labels are provided, use them for computing loss
        compute_loss = True if y is not None else False

        if compute_loss:
            stop_limit = len(y)-1
            # get starting word to initialize decoder
            curr_word = y[0]
        else:
            # intialize starting word to GO_ID symbol
            curr_word = Variable(xp.full((batch_size,), GO_ID, dtype=xp.int32))

        # flag to track if all sentences in batch have predicted EOS
        check_if_all_eos = xp.full((batch_size,), False, dtype=xp.bool_)

        while npred < (stop_limit):
            # encode tokens
            pred_out = self.decode(curr_word)
            pred_word = F.argmax(pred_out, axis=1)

            # save prediction at this time step
            if npred == 0:
                pred_sents = pred_word.data
            else:
                pred_sents = xp.vstack((pred_sents, pred_word.data))

            if compute_loss:
                # compute loss
                # softmax not required to select the most probable output
                # of the decoder
                # softmax only required if sampling from the predicted
                # distribution
                loss += F.softmax_cross_entropy(pred_out, y[npred+1],
                                                   class_weight=self.mask_pad_id)
                # uncomment following line if labeled data to be 
                # used at next time step
                # curr_word = y[npred+1]
            # else:
            curr_word = pred_word

            # check if EOS is predicted for all sentences
            # exit function if True
            check_if_all_eos[pred_word.data == EOS_ID] = True
            if xp.all(check_if_all_eos == EOS_ID):
                break
            # increment number of predictions made
            npred += 1

        return pred_sents.T, loss


    def forward_cnn(self, X):
        # perform convolutions
        h = F.relu(self[self.cnns[0]](X))

        for i in range(len(self.cnns[1:])):
            h = F.concat((h, F.relu(self[self.cnns[i]](X))), axis=1)

        # max pooling
        h = F.max_pooling_nd(h, ksize=max_pool_stride,
                             stride=max_pool_stride,
                             pad=max_pool_pad)

        # out dimension:
        # batch size * cnn out dim * num time frames after pooling
        return h

    def forward_highway(self, X):
        # highway
        for i in range(len(self.highway)):
            h = self[self.highway[i]](X)
            # print(h.shape)
        return h

    def forward_rnn(self, X):
        self.reset_state()
        in_size, batch_size, in_dim = X.shape
        # print("X", X.shape)
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

    def forward_enc(self, X):
        if MODEL_TYPE == MODEL_CNN:
            h = self.forward_cnn(X)
            h = F.rollaxis(h, 2)
        _, _ = self.forward_rnn(h)


    def forward(self, X, y=None):
        # get shape
        batch_size, in_dim, in_num_steps = X.shape
        # encode input
        self.forward_enc(X)
        # initialize decoder LSTM to final encoder state
        self.set_decoder_state()
        # swap axes of the decoder batch
        if y is not None:
            y = F.swapaxes(y, 0, 1)
        # check if train or test
        if chainer.config.train:
            # decode
            self.loss = self.decode_batch(y)
            # consistent return statement
            return [], self.loss
        else:
            # predict
            return self.predict_batch(batch_size=batch_size, pred_limit=MAX_EN_LEN, y=y)

# In[ ]:

