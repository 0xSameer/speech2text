# coding: utf-8

from basics import *
from nn_config import *


class SpeechEncoderDecoder(Chain):

    def __init__(self, gpunum):
        '''
        n_speech_dim: dimensions for the speech features
        vsize:   vocabulary size
        nlayers: # layers
        attn:    if True, use attention
        '''
        # Store GPU id
        self.gpuid = gpunum

        self.init_params()

        if self.gpuid >= 0:
            cuda.get_device(self.gpuid).use()

        super(SpeechEncoderDecoder, self).__init__()

        self.init_model()


    def init_params(self):

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

        self.vocab_size_es = vocab_size_es

        self.vocab_size_en = vocab_size_en

        self.max_pool_stride = max_pool_stride
        self.max_pool_pad = max_pool_pad


    def add_rnn_layers(self, layer_names, in_units, out_units, scale):
        w = chainer.initializers.HeNormal()
        # add first layer
        self.add_link(layer_names[0], self.RNN(in_units, out_units))
        if USE_LN:
            self.add_link("{0:s}_ln".format(layer_names[0]), L.LayerNormalization(out_units))
        # add remaining layers
        for rnn_name in layer_names[1:]:
            self.add_link(rnn_name, self.RNN(out_units*scale, out_units))
            if USE_LN:
                self.add_link("{0:s}_ln".format(rnn_name), L.LayerNormalization(out_units))


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
        # self.rnn_rev_enc = ["L{0:d}_rev_enc".format(i) for i in range(num_layers_enc)]
        # self.add_rnn_layers(self.rnn_rev_enc,
        #                     in_dim,
        #                     self.n_units,
        #                     scale=scale)

        # add LSTM layers
        # first layer appends previous ht, and therefore,
        # in_units = embed units + hidden units
        self.rnn_dec = ["L{0:d}_dec".format(i) for i in range(num_layers_dec)]
        # self.add_rnn_layers(self.rnn_dec,
        #                     self.embed_units,
        #                     2*self.n_units,
        #                     scale=scale)
        self.add_rnn_layers(self.rnn_dec,
                            self.embed_units+(2*self.n_units),
                            self.n_units,
                            scale=scale)

    def init_cnn_model(self):
        w = chainer.initializers.HeNormal()
        self.cnns = []
        # add CNN layers
        cnn_out_dim = 0
        if len(cnn_filters) > 0:
            for l in cnn_filters:
                lname = "CNN_{0:d}".format(l['ksize'])
                cnn_out_dim += l["out_channels"]
                self.cnns.append(lname)
                self.add_link(lname, L.ConvolutionND(**l, initialW=w))

            self.cnn_out_dim = cnn_out_dim

            if USE_BN:
                # add batch normalization layer on the output of the conv layer
                self.add_link('cnn_bn', L.BatchNormalization(self.cnn_out_dim))

            # add highway layers
            self.highway = ["highway_{0:d}".format(i)
                             for i in range(num_highway_layers)]
            for hname in self.highway:
                self.add_link(hname, L.Highway(self.cnn_out_dim))
        else:
            self.cnn_out_dim = CNN_IN_DIM
    # end init_cnn_model()

    def init_deep_cnn_model(self):
        w = chainer.initializers.HeNormal()
        self.cnns = []
        # add CNN layers
        cnn_out_dim = 0
        reduce_dim = CNN_IN_DIM
        if CNN_TYPE == DEEP_2D_CNN:
            reduce_dim = reduce_dim // 1
        if len(cnn_filters) > 0:
            for i, l in enumerate(cnn_filters):
                lname = "CNN_{0:d}".format(i)
                cnn_out_dim += l["out_channels"]
                self.cnns.append(lname)
                if CNN_TYPE == DEEP_1D_CNN:
                    self.add_link(lname, L.ConvolutionND(**l, initialW=w))
                else:
                    self.add_link(lname, L.Convolution2D(**l, initialW=w))
                    reduce_dim = math.ceil(reduce_dim / l["stride"][1])
                if USE_BN:
                    self.add_link('{0:s}_bn'.format(lname), L.BatchNormalization((l["out_channels"])))

            self.cnn_out_dim = cnn_filters[-1]["out_channels"]
            if CNN_TYPE == DEEP_2D_CNN:
                self.cnn_out_dim *= reduce_dim

            # add highway layers
            self.highway = ["highway_{0:d}".format(i)
                             for i in range(num_highway_layers)]
            for hname in self.highway:
                self.add_link(hname, L.Highway(self.cnn_out_dim))
        else:
            self.cnn_out_dim = CNN_IN_DIM
    # end init_deep_cnn_model()


    def init_model(self):

        if enc_key != 'sp':
            # add embedding layer
            self.add_link("embed_enc", L.EmbedID(self.vocab_size_es,
                                                 self.embed_units))

        self.scale = 1

        if CNN_TYPE == SINGLE_1D_CNN:
            self.init_cnn_model()
        else:
            self.init_deep_cnn_model()
        rnn_in_units = self.cnn_out_dim

        # initialize RNN layers
        print("cnn_out_dim = rnn_in_units = ", rnn_in_units)
        self.init_rnn_model(self.scale, rnn_in_units)

        # add embedding layer
        self.add_link("embed_dec", L.EmbedID(self.vocab_size_en,
                                             self.embed_units))

        if self.attn > 0:
            # add context layer for attention
            # self.add_link("context", L.Linear(4*self.n_units, 2*self.n_units))
            # if USE_BN:
            #     self.add_link("context_bn",
            #                    L.BatchNormalization(2*self.n_units))
            if ATTN_W:
                self.add_link("attn_Wa", L.Linear(self.n_units, self.n_units))
            self.add_link("context", L.Linear(2*self.n_units, 2*self.n_units))
            # if USE_BN:
            #     self.add_link("context_bn",
            #                    L.BatchNormalization(2*self.n_units))

        # add output layer
        self.add_link("out", L.Linear(2*self.n_units, vocab_size_en))

        # create masking array for pad id
        with cupy.cuda.Device(self.gpuid):
            self.mask_pad_id = xp.ones(self.vocab_size_en, dtype=xp.float32)
        # make the class weight for pad id equal to 0
        # this way loss will not be computed for this predicted loss
        self.mask_pad_id[0] = 0


    def reset_state(self):
        # reset the state of LSTM layers
        # for rnn_name in self.rnn_enc + self.rnn_rev_enc + self.rnn_dec:
        for rnn_name in self.rnn_enc + self.rnn_dec:
            self[rnn_name].reset_state()
        self.loss = 0

    def set_decoder_state(self):
        # set the hidden and cell state (if LSTM) of the first RNN in the decoder

        # concatenate cell state of both enc LSTMs
        # c_state = F.concat((self[self.rnn_enc[-1]].c, self[self.rnn_rev_enc[-1]].c))
        # concatenate hidden state of both enc LSTMs
        # h_state = F.concat((self[self.rnn_enc[-1]].h, self[self.rnn_rev_enc[-1]].h))

        for enc_name, dec_name in zip(self.rnn_enc, self.rnn_dec):
            if self.lstm1_or_gru0:
                self[dec_name].set_state(self[enc_name].c, self[enc_name].h)
            else:
                self[dec_name].set_state(self[enc_name].h)


    def compute_context_vector(self, dec_h):
        batch_size, n_units = dec_h.shape
        # attention weights for the hidden states of each word in the input list

        if ATTN_W:
            # learnable parameters
            ht = self.attn_Wa(dec_h)
        else:
            # dot product attention
            ht = dec_h

        weights = F.batch_matmul(self.enc_states, ht)

        # '''
        # this line is valid when no max pooling or sequence length manipulation is performed
        # weights = F.where(self.mask, weights, self.minf)
            # '''

        alphas = F.softmax(weights)
        # compute context vector
        cv = F.squeeze(F.batch_matmul(F.swapaxes(self.enc_states, 2, 1), alphas), axis=2)

        return cv, alphas


    def feed_rnn(self, rnn_in, rnn_layers, highway_layers=None):
        # feed into first rnn layer
        hs = rnn_in
        # feed into remaining rnn layers
        for rnn_layer in rnn_layers:
            if USE_DROPOUT:
                hs = F.dropout(self[rnn_layer](hs), ratio=DROPOUT_RATIO)
            else:
                hs = self[rnn_layer](hs)

            if USE_LN:
                bn_name = "{0:s}_ln".format(rnn_layer)
                hs = self[bn_name](hs)
        return hs

    def encode(self, data_in, rnn_layers):
        # if cnn + highways used
        if num_highway_layers > 0:
            h = self.forward_highway(data_in)
            h = self.feed_rnn(h, rnn_layers)
        else:
            h = self.feed_rnn(data_in, rnn_layers)
        # return F.relu(h)
        return h

    def decode(self, word, ht):
        if USE_DROPOUT:
            embed_id = F.dropout(self.embed_dec(word),DROPOUT_RATIO)
        else:
            embed_id = self.embed_dec(word)
        rnn_in = F.concat((embed_id, ht), axis=1)
        h = self.feed_rnn(rnn_in, self.rnn_dec)

        cv, _ = self.compute_context_vector(h)
        cv_hdec = F.concat((cv, h), axis=1)
        ht = self.context(cv_hdec)
        # batch normalization before non-linearity
        # if USE_BN:
        #     ht = self.context_bn(ht)
        ht = F.tanh(ht)

        predicted_out = self.out(ht)

        return predicted_out, ht

    def decode_batch(self, decoder_batch):
        batch_size = decoder_batch.shape[1]
        loss = 0
        ht = Variable(xp.zeros((batch_size, 2*self.n_units), dtype=xp.float32))

        decoder_input = decoder_batch[0]

        # for all sequences in the batch, feed the characters one by one
        for curr_word, next_word in zip(decoder_batch, decoder_batch[1:]):

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                decoder_input = curr_word
            # else:
            #     decoder_input = F.argmax(predicted_out, axis=1)

            # encode tokens

            predicted_out, ht = self.decode(decoder_input, ht)

            decoder_input = F.argmax(predicted_out, axis=1)

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
        with cupy.cuda.Device(self.gpuid):
            check_if_all_eos = xp.full((batch_size,), False, dtype=xp.bool_)

        ht = Variable(xp.zeros((batch_size, 2*self.n_units), dtype=xp.float32))

        while npred < (stop_limit):
            # encode tokens
            pred_out, ht = self.decode(curr_word, ht)
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
        h = F.swapaxes(X,1,2)
        h = F.relu(self[self.cnns[0]](h))

        for i in range(len(self.cnns[1:])):
            h = F.concat((h, F.relu(self[self.cnns[i]](X))), axis=1)

        # max pooling
        # h = F.max_pooling_nd(h, ksize=max_pool_stride,
        #                      stride=max_pool_stride,
        #                      pad=max_pool_pad)

        # batch normalization
        if USE_BN:
            h = self.cnn_bn(h, finetune=FINE_TUNE)

        # out dimension:
        # batch size * num time frames after pooling * cnn out dim
        h = F.rollaxis(h, 2)
        return h
    # end forward_cnn()

    def forward_deep_cnn(self, h):
        # check and prepare for 2d convolutions
        if CNN_TYPE == DEEP_2D_CNN:
            h = F.expand_dims(h, 2)
            # h = F.reshape(h, (h.shape[:2] + tuple([-1,SPEECH_DIM // 3])))
        h = F.swapaxes(h,1,2)

        for i, cnn_layer in enumerate(self.cnns):
            h = self[cnn_layer](h)
            # h = F.max_pooling_nd(h, ksize=cnn_max_pool[i],
            #                      stride=cnn_max_pool[i],
            #                      pad=max_pool_pad)
            # batch normalization before non-linearity
            if USE_BN:
                bn_lname = '{0:s}_bn'.format(cnn_layer)
                h = self[bn_lname](h)
            h = F.relu(h)

        # out dimension:
        # batch size * num time frames after pooling * cnn out dim
        if CNN_TYPE == DEEP_2D_CNN:
            h = F.swapaxes(h,1,2)
            h = F.reshape(h, h.shape[:2] + tuple([-1]))
            h = F.rollaxis(h,1)
        else:
            h = F.rollaxis(h, 2)
        return h

    def forward_highway(self, X):
        # highway
        for i in range(len(self.highway)):
            if USE_DROPOUT:
                h = F.dropout(self[self.highway[i]](X), ratio=DROPOUT_RATIO)
                # if USE_BN:
                #     h = self[self.highway_bn[i]](h, finetune=FINE_TUNE)
            else:
                h = self[self.highway[i]](X)
                # if USE_BN:
                #     h = self[self.highway_bn[i]](h, finetune=FINE_TUNE)
        return h

    def forward_rnn(self, X):
        self.reset_state()
        in_size, batch_size, in_dim = X.shape
        for i in range(in_size):
            if i > 0:
                h_fwd = F.concat((h_fwd,
                                  F.expand_dims(self.encode(X[i],
                                    self.rnn_enc), 0)),
                                  axis=0)
                # h_rev = F.concat((h_rev,
                #                   F.expand_dims(self.encode(X[-i],
                #                     self.rnn_rev_enc), 0)),
                #                   axis=0)
            else:
                h_fwd = F.expand_dims(self.encode(X[i], self.rnn_enc), 0)
                # h_rev = F.expand_dims(self.encode(X[-i], self.rnn_rev_enc), 0)

        # h_rev = F.flipud(h_rev)
        # self.enc_states = F.concat((h_fwd, h_rev), axis=2)
        self.enc_states = h_fwd
        self.enc_states = F.swapaxes(self.enc_states, 0, 1)
        # return h_fwd, h_rev

    def forward_enc(self, X):
        if enc_key != 'sp':
            h = self.embed_enc(X)
        else:
            h = X
        if len(self.cnns) > 0:
            if CNN_TYPE == SINGLE_1D_CNN:
                h = self.forward_cnn(h)
            else:
                h = self.forward_deep_cnn(h)
        self.forward_rnn(h)


    def forward(self, X, y=None):
        # get shape
        batch_size = X.shape[0]
        # check whether to add noi, start=1se
        if ADD_NOISE and chainer.config.train:
            # due to CUDA issues with random number generator
            # creating a numpy array and moving to GPU
            noise = Variable(np.random.normal(1.0,
                            NOISE_STDEV,
                            size=X.shape).astype(np.float32))
            if gpuid >= 0:
                noise.to_gpu(gpuid)
            X = X * noise
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
            # self.enc_states = []
            # consistent return statement
            return [], self.loss
        else:
            # predict
            return(self.predict_batch(batch_size=batch_size, pred_limit=MAX_EN_LEN, y=y))
            # consistent return statement

# In[ ]:

