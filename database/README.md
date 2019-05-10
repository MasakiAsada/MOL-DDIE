# Usage

## download drugbank xml file
Create drugbank account and download drugbank xml file.
Set a path `export DRUGBANK_XML=your drugbank file`.

## preprocess
```
cd preprocessors
python3 get_dict.py $DRUGBANK_XML ../dicts/name_dict.pkl ../dicts/smiles_dict.pkl
python3 entity2dbid.py ../dicts/name_dict.pkl ../../corpus/semeval2013/instances/train
python3 entity2dbid.py ../dicts/name_dict.pkl ../../corpus/semeval2013/instances/test
python3 xml2inst.py $DRUGBANK_XML ../dicts/smiles_dict.pkl ../../corpus/semeval2013/instances/test
cd ..
```

## train GCN
```
cd src
python3 run.py ../yamls/drugbank_parameter.yaml
```

## make molecular vector table
```
python3 pred.py ../yamls/drugbank_parameter.yaml
```
