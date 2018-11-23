import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
cp = cuda.cupy


class CNN(chainer.Chain):
    def __init__(self, args, w_v):
        super(CNN, self).__init__()
        with self.init_scope():
            self.emb = L.EmbedID(w_v.shape[0], w_v.shape[1], initialW=w_v)
            p_rng = args.posemb_range
            init_pos_w = np.random.uniform(low=-1*p_rng, high=p_rng, size=[2*args.max_sent_len, args.posemb_size])
            if args.posemb_size > 0:
                self.posemb = L.EmbedID(2*args.max_sent_len, args.posemb_size, initialW=init_pos_w)
            self.conv_lst = chainer.ChainList()
            for x in args.win_size:
                self.conv_lst.add_link(L.Convolution2D(1, args.n_filter,
                    ksize=(x, w_v.shape[1] + 2*args.posemb_size),
                    pad=((x - 1) // 2, 0)))

        self.max_sent_len = args.max_sent_len
        self.emb_size = w_v.shape[1]
        self.posemb_size = args.posemb_size
        self.win_size = args.win_size
        self.n_filter = args.n_filter
        self.attention = args.attention
        self.a_dr = args.a_dr

    def __call__(self, x, ep, mask):
        # Word embedding
        we = self.emb(x)

        mask = F.expand_dims(mask, axis=-1)

        # Position embedding
        if self.posemb_size > 0:
            serial_num = F.expand_dims(cp.arange(self.max_sent_len, 2*self.max_sent_len), axis=0)
            bep, bserial_num = F.broadcast(F.expand_dims(ep, axis=-1), serial_num)
            rd = bserial_num - bep
            rd1, rd2 = rd[:, 0, :], rd[:, 1, :]
            pe1, pe2 = self.posemb(rd1), self.posemb(rd2)
            we = F.concat((we, pe1, pe2), axis=2)

        self.ep = ep
        _, bmask = F.broadcast(e, mask)
        
        # Masking (multiply 0 by padding terms)
        h = e * bmask

        # Convolution
        h = F.expand_dims(h, axis=1)
        h_conv_lst = []
        for link in self.conv_lst:
            h_conv = F.relu(link(h))
            h_pool = F.max_pooling_2d(h_conv, ksize=(self.max_sent_len, 1))
            h_pool = F.reshape(h_pool, (-1, self.n_filter))
            h_conv_lst.append(h_pool)
        h_pool_concat = F.concat(h_conv_lst, axis=1)

        return h_pool_concat
