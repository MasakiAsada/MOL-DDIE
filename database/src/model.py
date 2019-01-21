from rdkit import Chem
import chainer
import chainer.links as L
import chainer.functions as F
from chainer_chemistry.models import NFP, GGNN


class DDIBinaryClassifier(chainer.Chain):
    def __init__(self, params):
        super(DDIBinaryClassifier, self).__init__()
        with self.init_scope():
            if params['gcn_method'] == 'nfp':
                self.gcn = NFP(out_dim=params['gcn_out_dim'], hidden_dim=params['gcn_hidden_dim'])
            elif params['gcn_method'] == 'ggnn':
                self.gcn = GGNN(out_dim=params['gcn_out_dim'], hidden_dim=params['gcn_hidden_dim'])
            self.hidden_dim = params['hidden_dim']

            self.l1 = L.Linear(self.gcn.out_dim * 2, self.hidden_dim)
            self.l2 = L.Linear(self.hidden_dim, 2)

    def __call__(self, atoms1, adjs1, atoms2, adjs2):
        x1 = self.gcn(atoms1, adjs1)
        x2 = self.gcn(atoms2, adjs2)
        h = F.concat((x1, x2), axis=1)
        
        h = F.relu(self.l1(h))
        y = self.l2(h)

        return y, x1, x2
