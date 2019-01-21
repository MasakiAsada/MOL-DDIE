mkdir train
mkdir dev
ratio=4
n_db=`ls -1 $SEMEVAL2013_TRAIN/DrugBank/*.xml | wc -l`
echo $n_db
n_db_tr=$((n_db / (ratio+1) * ratio))
echo $n_db_tr

count=0
for i in all/DrugBank/*.xml
do
    count=$((count+1))
    if [ $count -lt $n_db_tr ]; then
        cp $i train
    else
        cp $i dev
    fi
done

n_ml=`ls -1 $SEMEVAL2013_TRAIN/MedLine/*.xml | wc -l`
echo $n_ml
n_ml_tr=$((n_ml / (ratio+1) * ratio))
echo $n_ml_tr

count=0
for i in all/MedLine/*.xml
do
    count=$((count+1))
    if [ $count -lt $n_ml_tr ]; then
        cp $i train
    else
        cp $i dev
    fi
done
