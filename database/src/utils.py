import numpy as np
from rdkit import Chem
from chainer import cuda
cp = cuda.cupy
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict

def get_smiles(data, smiles_dict):
    smiles1 = []
    smiles2 = []
    label = []
    for l in data:
        x, y = l.split('\t')
        x1, x2 = x.split(':')
        smiles1.append(smiles_dict[x1])
        smiles2.append(smiles_dict[x2])
        label.append(int(y))
    return np.array(smiles1), np.array(smiles2), np.array(label).astype('i')
    
def get_max_n_atoms(data):
    mols = [Chem.MolFromSmiles(x) for x in data]
    n_atoms = [int(x.GetNumAtoms()) for x in mols]
    return max(n_atoms)

def prep_data(smiles, max_n_atom, method='nfp'):
    mol = [Chem.MolFromSmiles(x) for x in smiles]
    preprocessor = preprocess_method_dict[method](out_size=max_n_atom)
    atoms = cp.array([preprocessor.get_input_features(x)[0] for x in mol])
    adjs = cp.array([preprocessor.get_input_features(x)[1] for x in mol])
    return atoms, adjs

def calculate_accuracy(pred, gold):
    n = len(pred)
    true_cnt = 0
    for p, g in zip(pred, gold):
        if p == g:
            true_cnt += 1
    return true_cnt / n

