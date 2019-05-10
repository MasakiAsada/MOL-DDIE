# Usage

## download dataset
Ask task organizers and download datasets.
Set paths, for exmaple, `export SEMEVAL_TRAIN_DIR=your train dir`, `export SEMEVAL_TEST_DIR=your test dir`.

## replace space to underscore
```
for i in ${SEMEVAL_TRAIN_DIR}/*.xml; mv $i `echo $i | sed -e "s/\ /_/g"`
for i in ${SEMEVAL_TEST_DIR}/*.xml; mv $i `echo $i | sed -e "s/\ /_/g"`
```

## convert xml to brat format
```
mkdir brat_train brat_test
for i in ${SEMEVAL_TRAIN_DIR}/*.xml; python3 xml2brat.py $i brat_train/`basename $i .xml`
for i in ${SEMEVAL_TEST_DIR}/*.xml; python3 xml2brat.py $i brat_test/`basename $i .xml`
```

## convert brat format to instance
```
mkdir instances
python3 brat2inst.py brat_train instances/train
python3 brat2inst.py brat_test instances/test
```

## split into words using GENIA tagger
Download GENIA tagger from http://www.nactem.ac.uk/GENIA/tagger/
```
cd geniatagger
./geniatagger ../instances/train.sent > ../instances/train.split.sent
./geniatagger ../instances/test.sent > ../instances/test.split.sent
cd ..
```
