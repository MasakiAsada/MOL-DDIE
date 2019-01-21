import chainer
import chainer.links as L
import chainer.functions as F

from cnn import CNN

class RelationExtractor(chainer.Chain):
    def __init__(self, params, w2v, mol2v):
        super(RelationExtractor, self).__init__()
        with self.init_scope():
            self.cnn = CNN(params, w2v)
            self.hidden_dim = params['hidden_dim']
            self.out_dim = params['out_dim']
            self.mol2v = mol2v
            if mol2v is None:
                self.molemb_size = 0
            else:
                self.molemb_size = mol2v.shape[1]
                self.molemb = L.EmbedID(mol2v.shape[0], mol2v.shape[1], initialW=mol2v)
            self.l1 = L.Linear(len(self.cnn.window_size)*self.cnn.n_filter+2*self.molemb_size, self.hidden_dim)
            self.l2 = L.Linear(self.hidden_dim, self.out_dim)
            
    def __call__(self, x):
        h_conv = self.cnn(x)
        if self.mol2v is None:
            h = h_conv
        else:
            msl = self.cnn.max_sent_len
            mole1 = self.molemb(x[:, 3*msl])
            mole2 = self.molemb(x[:, 3*msl+1])
            mole1 = F.normalize(mole1)
            mole2 = F.normalize(mole2)
            h = F.concat((h_conv, mole1, mole2), axis=1)
        y = self.l2(F.relu(self.l1(h)))
            
        return y

    def init_params(self):
        self.update_cnt = 0
        for x in self.params():
            x.data *= 0

    def store_params(self, model):
        self.update_cnt += 1
        for x, y in zip(self.params(), model.params()):
            x.data += y.data

    def average_params(self, model):
        for x, y in zip(self.params(), model.params()):
            x.data = y.data / model.update_cnt
