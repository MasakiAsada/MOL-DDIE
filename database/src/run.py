import sys
import time
import yaml
import pickle as pkl
import numpy as np

from rdkit import Chem
import chainer
import chainer.functions as F
from chainer import optimizers, cuda, Variable, serializers
cp = cuda.cupy

from utils import get_smiles, get_max_n_atoms, prep_data, calculate_accuracy
from model import DDIBinaryClassifier


if len(sys.argv) != 2:
    sys.stderr.write('Usage: python3 {} yamlfile'.format(sys.argv[0]))
    sys.exit(-1)

with open(sys.argv[1], 'r') as f:
    params = yaml.load(f)

with open(params['smiles_dict_path'], 'rb') as f:
    smiles_dict = pkl.load(f)

with open(params['train_path'], 'r') as f:
    train = f.read().strip().split('\n')
with open(params['test_path'], 'r') as f:
    test = f.read().strip().split('\n')
print('n_train={} n_test={}'.format(len(train), len(test)))
smiles1tr, smiles2tr, labeltr = get_smiles(train, smiles_dict)
smiles1te, smiles2te, labelte = get_smiles(test, smiles_dict)

model = DDIBinaryClassifier(params)
model.to_gpu()
optimizer = optimizers.Adam()
optimizer.setup(model)

max_n_atoms = get_max_n_atoms(list(smiles_dict.values()))
method = params['gcn_method']
bs = params['batchsize']

def train(X1, X2, Y):
    start = time.time()
    n = len(X1)
    sffindx = np.random.permutation(n)
    losses = 0
    for i in range(0, n, bs):
        x1 = X1[sffindx[i:(i+bs) if (i+bs) < n else n]]
        x2 = X2[sffindx[i:(i+bs) if (i+bs) < n else n]]
        atoms1, adjs1 = prep_data(x1, max_n_atoms, method)
        atoms2, adjs2 = prep_data(x2, max_n_atoms, method)
        y = cp.array(Y[sffindx[i:(i+bs) if (i+bs) < n else n]])
        model.zerograds()
        p, _, _ = model(atoms1, adjs1, atoms2, adjs2)
        loss = F.softmax_cross_entropy(p, y)
        loss.backward()
        optimizer.update()
        losses += cuda.to_cpu(loss.data)
    print('Train: elapsedtime={:.2f} loss={:.2f}'.format(time.time()-start, losses))

def test(X1, X2, Y):
    start = time.time()
    n = len(X1)
    losses = 0
    P = np.array([])
    with chainer.using_config('train', False):
        for i in range(0, n, bs):
            x1 = X1[i:(i+bs) if (i+bs) < n else n]
            x2 = X2[i:(i+bs) if (i+bs) < n else n]
            atoms1, adjs1 = prep_data(x1, max_n_atoms, method)
            atoms2, adjs2 = prep_data(x2, max_n_atoms, method)
            y = cp.array(Y[i:(i+bs) if (i+bs) < n else n])
            p, _, _ = model(atoms1, adjs1, atoms2, adjs2)
            loss = F.softmax_cross_entropy(p, y)
            losses += cuda.to_cpu(loss.data)
            pred = F.argmax(p, axis=1)
            P = np.concatenate((P, cuda.to_cpu(pred.data)))
    acc = calculate_accuracy(P, Y)
    print('Test : elapsedtime={:.2f} loss={:.2f}'.format(time.time()-start, losses))
    print('  Accuracy={:.4f}'.format(acc))

for epoch in range(1, params['n_epoch']+1):
    print('epoch={}'.format(epoch))
    train(smiles1tr, smiles2tr, labeltr)
    test(smiles1te, smiles2te, labelte)

serializers.save_npz(params['model_path'], model)
