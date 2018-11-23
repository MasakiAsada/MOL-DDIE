import chainer
import chainer.links as L
from chainer import optimizers, cuda, Variable, serializers
import chainer.functions as F
cp = cuda.cupy
from chainer_chemistry.models import NFP, GGNN, SchNet, WeaveNet
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict

from rdkit import Chem

import copy
import numpy as np
import time
import argparse

import preprocess
from cnn import CNN
from utils import count, calculate_microF

#path_tr = '../data/semeval/corpus/train.json'
#path_te = '../data/semeval/corpus/test.json'
path_tr = '../data/semeval/corpus/dev_train.json'
path_te = '../data/semeval/corpus/dev_dev.json'

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--max_sent_len', default=180, type=int, help='max length of sentence')
parser.add_argument('-b', '--batchsize', default=50, type=int, help='mini batch size of cnn')
parser.add_argument('-e', '--n_epoch', default=15, type=int, help='number of epoch')
parser.add_argument('-f', '--n_filter', default=100, type=int, help='number of convolution filter')
parser.add_argument('-w', '--win_size', nargs='*', default=[3,5,7], type=int, help='list of convolution window size')
parser.add_argument('-p', '--posemb_size', default=0, type=int, help='size of position embedding')
parser.add_argument('-pr', '--posemb_range', default=0.01, type=float, help='range of posemb')
parser.add_argument('-at', '--attention', default=0, type=int, help='input attention')
parser.add_argument('-dr', '--a_dr', default=0., type=float, help='attention of drug entity')
parser.add_argument('-av', '--average', default=1, type=int, help='average all parameters')
parser.add_argument('-r', '--learning_rate', default=0.001, type=float, help='learning rate')
parser.add_argument('-d', '--dropout', default=0., type=float, help='rate of dropout')
parser.add_argument('-l2', '--l2_reg', default=0.0001, type=float, help='rate of weight decay')
parser.add_argument('-m', '--mid_dim', default=500, type=int, help='dimention of middle layer')
parser.add_argument('-gm', '--gcn_method', default='no', type=str, help='gcn method')
arguments = parser.parse_args()
print(arguments)

class Extract_Relation(chainer.Chain):
    def __init__(self, cnn, mol_table):
        super(Extract_Relation, self).__init__()
        with self.init_scope():
            if mol_table is None:
                mol_shape = (0, 0)
            else:
                mol_shape = mol_table.shape
                self.mol_emb = L.EmbedID(mol_table.shape[0], mol_shape[1], initialW=mol_table)
            self.mol_table = mol_table
            self.cnn = cnn
            self.l1 = L.Linear(len(cnn.win_size) * cnn.n_filter + 2 * mol_shape[1], arguments.mid_dim)
            self.l2 = L.Linear(arguments.mid_dim, 5)
            
    def __call__(self, x, ep, mask):
        h_conv = F.dropout(self.cnn(x, ep, mask), ratio=arguments.dropout)
        h = h_conv
        y = self.l2(F.relu(self.l1(h)))
        return y

    def init_by_zeros(self):
        self.update_cnt = 0
        for x in self.namedparams():
            x[1].data *= 0

    def accumulate(self, model):
        self.update_cnt += 1
        for x in self.namedparams():
            for y in model.namedparams():
                if x[0] == y[0]:
                    x[1].data += y[1].data

    def divide(self, model):
        for x in self.namedparams():
            for y in model.namedparams():
                if x[0] == y[0]:
                    x[1].data = y[1].data / model.update_cnt

np.random.seed(1)
cp.random.seed(1)

tr, te, w_v = preprocess.main(path_tr, path_te, arguments)
if arguments.gcn_method == 'no':
    mol_table = None
model = Extract_Relation(CNN(arguments, w_v), mol_table)
model.to_gpu()
average_model = copy.deepcopy(model)
store_model = copy.deepcopy(model)
store_model.init_by_zeros()

optimizer = optimizers.Adam(arguments.learning_rate)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(arguments.l2_reg))

n = len(tr[0])
nt = len(te[0])
bs = arguments.batchsize

score_lst = []
for epoch in range(1, arguments.n_epoch+1):
    start = time.time()
    sffindx = np.random.permutation(n)
    total_loss = 0
    with chainer.using_config('cudnn_deterministic', True):
        for i in range(0, n, bs):
            model.zerograds()

            x = Variable(cp.array(tr[0][sffindx[i:(i+bs) if (i+bs) < n else n]]))
            t = Variable(cp.array(tr[1][sffindx[i:(i+bs) if (i+bs) < n else n]]))
            ep = Variable(cp.array(tr[2][sffindx[i:(i+bs) if (i+bs) < n else n]]))
            mask = Variable(cp.array(tr[3][sffindx[i:(i+bs) if (i+bs) < n else n]]))
            y = model(x, ep, mask)

            loss = F.softmax_cross_entropy(y, t)
            loss.backward()
            optimizer.update()
            total_loss += loss.data

            store_model.accumulate(model)

        #print('epoch %d elapsed_time %d' % (epoch, time.time()-start) )
        #print('  train loss %.4f' % (total_loss) )
        
        average_model.divide(store_model)
        r_d, p_d, p_n = 0, 0, 0
        with chainer.using_config('train', False):
            for i in range(0, nt, bs):
                xt = Variable(cp.array(te[0][i:(i+bs) if (i+bs) < nt else nt]))
                tt = Variable(cp.array(te[1][i:(i+bs) if (i+bs) < nt else nt]))
                ept = Variable(cp.array(te[2][i:(i+bs) if (i+bs) < nt else nt]))
                maskt = Variable(cp.array(te[3][i:(i+bs) if (i+bs) < nt else nt]))
                if arguments.average:
                    yt = average_model(xt, ept, maskt)
                else:
                    yt = model(xt, ept, maskt)
                r_d, p_d, p_n = count(F.argmax(yt, axis=1).data, tt.data, r_d, p_d, p_n)
        score = calculate_microF(r_d, p_d, p_n)
        print(score)
        score_lst.append(score)
score_lst.sort(key=lambda x:x[2])        
print('top', score_lst[-1][2])
k = 3
print('top{}_mean'.format(k), np.mean(np.array(score_lst)[-1*k:, 2]))
