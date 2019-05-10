mkdir train
mkdir dev
ratio=4
#n_db=`ls -1 ${SEMEVAL_TRAIN_DIR}/DrugBank/*.xml | wc -l`
n_db=`ls -1 Train/DrugBank/*.xml | wc -l`
echo $n_db
n_db_tr=$((n_db / (ratio+1) * ratio))
echo $n_db_tr

count=0
for i in Train/DrugBank/*.xml
do
    count=$((count+1))
    if [ $count -lt $n_db_tr ]; then
        cp $i train
    else
        cp $i dev
    fi
done

n_ml=`ls -1 Train/MedLine/*.xml | wc -l`
echo $n_ml
n_ml_tr=$((n_ml / (ratio+1) * ratio))
echo $n_ml_tr

count=0
for i in Train/MedLine/*.xml
do
    count=$((count+1))
    if [ $count -lt $n_ml_tr ]; then
        cp $i train
    else
        cp $i dev
    fi
done
