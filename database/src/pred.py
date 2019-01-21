import sys
import yaml
import pickle as pkl
import numpy as np

from rdkit import Chem
import chainer
from chainer import optimizers, cuda, serializers

from model import DDIBinaryClassifier
from utils prep_data, get_max_n_atoms

if len(sys.argv) != 2:
    sys.stderr.write('Usage: python3 {} yamlfile'.format(sys.argv[0]))
    sys.exit(-1)

with open(sys.argv[1], 'r') as f:
    params = yaml.load(f)

with open(params['smiles_dict_path'], 'rb') as f:
    smiles_dict = pkl.load(f)

dbid_dict = {}
smiles = []
for i, (k, v) in enumerate(smiles_dict.items()):
    dbid_dict[k] = i
    smiles.append(v)
smiles = np.array(smiles)

model = DDIBinaryClassifier(params)
model.to_gpu()
serializers.load_npz(params['model_path'], model)
optimizer = optimizers.Adam()
optimizer.setup(model)

max_n_atoms = get_max_n_atoms(list(smiles_dict.values()))
method = params['gcn_method']
bs = params['batchsize']

n = len(smiles)
Vec = np.zeros((0, model.gcn.out_dim))
for i in range(0, n, bs):
    x = smiles[i:(i+bs) if (i+bs) < n else n]
    atoms, adjs = prep_data(x, max_n_atoms, method)
    _, vec1, _ = model(atoms, adjs, atoms, adjs)
    Vec = np.concatenate((Vec, cuda.to_cpu(vec1.data)), axis=0)

with open(params['dbid_dict_path'], 'wb') as f:
    pkl.dump(dbid_dict, f)
np.save(params['mol2v_path'], Vec)
