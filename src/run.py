import sys
import copy
import numpy as np
import time
import pickle as pkl
import yaml

import chainer
from chainer import optimizers, cuda, serializers
import chainer.functions as F
cp = cuda.cupy

from preprocess import to_indx
from model import RelationExtractor
from cnn import CNN
from utils import calculate_microF


if len(sys.argv) != 2:
    sys.stderr.write('Usage: python3 %s yamlfile' % (sys.argv[0]))
    sys.exit(-1)

with open(sys.argv[1], 'r') as f:
    params = yaml.load(f)

word_indx = {}
label_indx = {}
Xtr, Ytr, w2v, mol2v = to_indx(params['train_path'], params, word_indx, label_indx, training=True)
Xte, Yte, _, _ = to_indx(params['test_path'], params, word_indx, label_indx, training=False)

params['out_dim'] = len(label_indx)
model = RelationExtractor(params, w2v, mol2v)
model.to_gpu()
average_model = copy.deepcopy(model)
store_model = copy.deepcopy(model)
store_model.init_params()

optimizer = optimizers.Adam(params['learning_rate'])
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(params['l2_lambda']))
if mol2v is not None: model.molemb.disable_update()
bs = params['batchsize']

def train(X, Y):
    start = time.time()
    losses = 0
    n = len(X)
    sffindx = np.random.permutation(n)
    for i in range(0, n, bs):
        model.zerograds()
        x = cp.array(X[sffindx[i:(i+bs) if (i+bs) < n else n]])
        y = cp.array(Y[sffindx[i:(i+bs) if (i+bs) < n else n]])
        p = model(x)
        loss = F.softmax_cross_entropy(p, y)
        loss.backward()
        optimizer.update()
        losses += cuda.to_cpu(loss.data)
        store_model.store_params(model)

    print('Train: elapsedtime={:.2f} loss={:.2f}'.format(time.time()-start, losses))

def test(X, Y):
    start = time.time()
    losses = 0
    n = len(X)
    P = np.array([])
    average_model.average_params(store_model)
    with chainer.using_config('train', False):
        for i in range(0, n, bs):
            x = cp.array(X[i:(i+bs) if (i+bs) < n else n])
            y = cp.array(Y[i:(i+bs) if (i+bs) < n else n])
            if params['averaging']:
                p = average_model(x)
            else:
                p = model(x)
            loss = F.softmax_cross_entropy(p, y)
            losses += cuda.to_cpu(loss.data)
            pred = F.argmax(p, axis=1)
            P = np.concatenate((P, cuda.to_cpu(pred.data)))

    print('Test : elapsedtime={:.2f} loss={:.2f}'.format(time.time()-start, losses))
    prec, recall, microF = calculate_microF(P, Y, label_indx['negative'])
    print('  Precision={:.4f} Recall={:.4f} microF={:.4f}'.format(prec, recall, microF))

    return P

for epoch in range(1, params['n_epoch']+1):
    print('epoch={}'.format(epoch))
    train(Xtr, Ytr)
    test(Xte, Yte)
