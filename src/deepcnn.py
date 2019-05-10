import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
cp = cuda.cupy


class DeepCNN(chainer.Chain):
    def __init__(self, params, w2v):
        super(DeepCNN, self).__init__()
        with self.init_scope():
            self.max_sent_len = params['max_sent_len']
            self.emb_size = w2v.shape[1]
            self.posemb_size = params['posemb_size']
            self.posemb_range = params['posemb_range']
            self.window_size = params['window_size']
            self.n_filter = params['n_filter']

            self.wemb = L.EmbedID(w2v.shape[0], w2v.shape[1], initialW=w2v)

            if self.posemb_size > 0:
                rng = self.posemb_range
                init_pos_w = np.random.uniform(low=-1*rng, high=rng,
                        size=[2*self.max_sent_len, self.posemb_size])
                self.posemb = L.EmbedID(2*self.max_sent_len, self.posemb_size, initialW=init_pos_w)

            self.convs = chainer.ChainList()
            for x in self.window_size:
                self.convs.add_link(L.ConvolutionND(ndim=1,
                    in_channels=self.emb_size+2*self.posemb_size,
                    out_channels=self.n_filter, ksize=x, pad=(x-1)//2))

            self.convs_ = chainer.ChainList()
            for x in self.window_size:
                self.convs_.add_link(L.ConvolutionND(ndim=1,
                    in_channels=3*self.n_filter,
                    out_channels=self.n_filter, ksize=x, pad=(x-1)//2))


    def __call__(self, x):
        msl = self.max_sent_len
        w = x[:, :msl]
        p1 = x[:, msl:2*msl]
        p2 = x[:, 2*msl:3*msl]

        # Word embedding
        we = self.wemb(w)

        # Position embedding
        if self.posemb_size > 0:
            pe1 = self.posemb(p1)
            pe2 = self.posemb(p2)
            emb = F.concat((we, pe1, pe2), axis=2)
        else:
            emb = we

        # Mask zero-padding elemets
        mask = cp.where(w > 0, 1, 0).astype('f')
        _, mask = F.broadcast(emb, F.expand_dims(mask, axis=-1))
        h = emb * mask

        # Convolution
        h = F.swapaxes(h, 1, 2)
        h_convs = []
        for link in self.convs:
            h_conv = F.relu(link(h))
            #h_conv = F.max_pooling_nd(h_conv, ksize=self.max_sent_len)
            h_conv = F.squeeze(h_conv)
            h_convs.append(h_conv)
        h_conv_concat = F.concat(h_convs, axis=1)

        n_layers = 3
        for i in range(n_layers):
            h_convs_ = []
            for link in self.convs_:
                h_conv_ = F.relu(link(h_conv_concat))
                #h_conv_ = F.max_pooling_nd(h_conv_, ksize=self.max_sent_len)
                h_conv_ = F.squeeze(h_conv_)
                h_convs_.append(h_conv_)
            h_conv_concat = F.concat(h_convs_, axis=1)

        h_convs_ = []
        for link in self.convs_:
            h_conv_ = F.relu(link(h_conv_concat))
            h_conv_ = F.max_pooling_nd(h_conv_, ksize=self.max_sent_len)
            h_conv_ = F.squeeze(h_conv_)
            h_convs_.append(h_conv_)
        h_conv_concat = F.concat(h_convs_, axis=1)

        return h_conv_concat
