 #coding: utf-8

from basics import *

class SpeechEncoderDecoder(Chain):
    def __init__(self, m_cfg, gpuid):
        self.m_cfg = m_cfg
        self.gpuid = gpuid

        self.init_params()

        if self.gpuid >= 0:
            cuda.get_device(self.gpuid).use()

        super(SpeechEncoderDecoder, self).__init__()
        self.init_model()

    def init_params(self):
        #----------------------------------------------------------------------
        # determine rnn type
        #----------------------------------------------------------------------
        if self.m_cfg['rnn_unit'] == RNN_GRU:
            self.RNN = L.GRU
        else:
            self.RNN = L.LSTM
        #----------------------------------------------------------------------
        # get vocab size
        #----------------------------------------------------------------------
        print("-"*80)
        v_path = os.path.join(self.m_cfg['data_path'], 
                              self.m_cfg["bagofwords_vocab"])
        vocab_dict = pickle.load(open(v_path, "rb"))
        self.v_size_es = 0
        self.v_size_en = len(vocab_dict['w2i'])
        print("bow vocab size = {0:d}".format(self.v_size_en))
        print("-"*80)
        #----------------------------------------------------------------------


    def init_deep_cnn_model(self):
        CNN_IN_DIM = self.m_cfg['sp_dim']
        # ---------------------------------------------------------------------
        # initialize list of cnn layers
        # ---------------------------------------------------------------------
        self.cnns = []
        # -----------------------------------------------------------------
        # using He initializer
        # -----------------------------------------------------------------
        # add CNN layers
        w = chainer.initializers.HeNormal()
        for i, l in enumerate(self.m_cfg['cnn_layers'][:-1]):
            lname = "CNN_{0:d}".format(i)
            self.cnns.append(lname)
            self.add_link(lname, L.Convolution2D(**l, initialW=w))
            if self.m_cfg['bn']:
                # ---------------------------------------------------------
                # add batch normalization
                # ---------------------------------------------------------
                self.add_link('{0:s}_bn'.format(lname), L.BatchNormalization((l["out_channels"])))
            # end batch normalization
        # end for - adding cnn layers
        l = self.m_cfg['cnn_layers'][-1]
        l['out_channels'] = self.v_size_en
        lname = "CNN_OUT"
        self.cnns.append(lname)
        self.add_link(lname, L.Convolution2D(**l, initialW=w))
        # -----------------------------------------------------------------
    # end init_deep_cnn_model

    def init_model(self):
        xp = cuda.cupy if self.gpuid >= 0 else np
        # ---------------------------------------------------------------------
        # add cnn layer
        # ---------------------------------------------------------------------
        self.init_deep_cnn_model()
        # h_units_0, h_units = self.cnn_out_dim, self.bag_size_en

        # # Add final prediction layer
        # self.add_link("out", L.Linear(h_units_0, h_units_0))
        # -----------------------------------------------------------------

    def forward_deep_cnn(self, h):
        # ---------------------------------------------------------------------
        # check and prepare for 2d convolutions
        # ---------------------------------------------------------------------
        h = F.expand_dims(h, 2)
        h = F.swapaxes(h,1,2)
        # ---------------------------------------------------------------------
        for i, cnn_layer in enumerate(self.cnns):
            # -----------------------------------------------------------------
            # apply cnn
            # -----------------------------------------------------------------
            h = self[cnn_layer](h)
            if "cnn_pool" in self.m_cfg:
                time_pool = self.m_cfg['cnn_pool'][i][0]
                if time_pool == -1:
                    time_pool = h.shape[-2]

                freq_pool = self.m_cfg['cnn_pool'][i][1]
                if freq_pool == -1:
                    freq_pool = h.shape[-1]
                h = F.max_pooling_nd(h, (time_pool, freq_pool))
            # -----------------------------------------------------------------
            # batch normalization before non-linearity
            # -----------------------------------------------------------------
            if self.m_cfg['bn']:
                bn_lname = '{0:s}_bn'.format(cnn_layer)
                h = self[bn_lname](h)
            # -----------------------------------------------------------------
            if i < (len(self.cnns)-1):
                h = F.relu(h)
            # -----------------------------------------------------------------
        h = F.squeeze(F.swapaxes(h,1,2))
        return h
    # -------------------------------------------------------------------------


    def compute_score(self, s):
        r = self.m_cfg["param_r"]
        S = (F.log(F.average(F.exp(r * s), axis=1)))/r
        return S

    def forward_bow(self, X, y=None, add_noise=0):
        # get shape
        batch_size = X.shape[0]
        # check whether to add noi, start=1se
        # ---------------------------------------------------------------------
        # check whether to add noise to speech input
        # ---------------------------------------------------------------------
        if add_noise > 0 and chainer.config.train:
            # due to CUDA issues with random number generator
            # creating a numpy array and moving to GPU
            noise = Variable(np.random.normal(1.0,
                                              add_noise,
                                              size=X.shape).astype(np.float32))
            if self.gpuid >= 0:
                noise.to_gpu(self.gpuid)
            X = X * noise
        # ---------------------------------------------------------------------
        # encode input
        h = self.forward_deep_cnn(X)
        # -----------------------------------------------------------------
        # Compute SCORE
        # -----------------------------------------------------------------
        S = self.compute_score(h)

        return(self.predict_bow_batch(S=S, y=y))
        # -----------------------------------------------------------------

    def predict_bow_batch(self, S, y=None):
        xp = cuda.cupy if self.gpuid >= 0 else np
        # to store loss
        loss = 0
        # if labels are provided, use them for computing loss
        compute_loss = True if y is not None else False
        pred_limit=self.m_cfg['max_en_pred']
        pred_words = []
        pred_probs = []
        # ---------------------------------------------------------------------
        predicted_out = F.sigmoid(S)
        for row in predicted_out.data:
            pred_inds = xp.where(row >= self.m_cfg["pred_thresh"])[0]
            if len(pred_inds) > pred_limit:
                pred_inds = xp.argsort(row)[-pred_limit:][::-1]
            #pred_words.append([bow_dict['i2w'][i] for i in pred_inds.tolist()])
            pred_words.append([i for i in pred_inds.tolist() if i > 2])
            np_row = xp.asnumpy(row)
            pred_probs.append(np_row)
        # -----------------------------------------------------------------
        if compute_loss:
            # compute loss
            loss = F.sigmoid_cross_entropy(S, y, reduce="mean")
        # -----------------------------------------------------------------
        return pred_words, loss, pred_probs

# In[ ]:

