# MOL-DDIE
Implementation of Enhancing Drug-Drug Interaction Extraction from Texts by Molecular Structure Information

# Requirements
python3
chainer >= 4
chainer_chemistry
rdkit
lxml
pyyaml
gensim


# Usage
## Preparation of the corpus sets
see [corpus/semeval2013/READEME.md](corpus/semeval2013/README.md)

## Preparation of the database
see [database/README.md](database/README.md)

## DDI Extraction
```
cd src
python3 run.py ../yamls/semeval2013_parameter.yaml
```
