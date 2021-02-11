#!/bin/bash
. ./path.sh

# Training (batch and gpu configs)
nj_init=2
nj_final=8
num_epochs=20
seed=0 # Useful for restarting with new seed
resume=
cmd="utils/retry.pl utils/queue.pl --mem 4G --gpu 1 --config conf/gpu.conf" 

. ./utils/parse_options.sh

echo "Num args: $#"

if [ $# -ne 2 ]; then
  echo "Usage: ./train_async_parallel.sh <train_script> <odir>"
  exit 1; 
fi

train_script=$1
odir=$2

mkdir -p ${odir}/log
# GPU vs. CPU training command

###############################################################################
# Training 
###############################################################################
echo""
echo "--------------- `date` -------------"
echo ""
# Train

resume_opts=
start_epoch=1
if [ ! -z $resume ]; then
  resume_opts="--resume ${resume}.mdl"
  start_epoch=$(( ${resume} + 1))
fi

[ -f ${odir}/.error ] && rm ${odir}/.error

for e in `seq ${start_epoch} ${num_epochs}`; do
  nj=`echo ${num_epochs} ${nj_final} ${nj_init} ${e} | awk '{print int($4*($2-$3)/$1) + $3}'`
  epoch_seed=`echo $nj $e $seed | awk '{print ($3+1)*$1*($2-1) + 1}'`
  (
    for j in `seq 1 ${nj}`; do
      job_seed=$(($epoch_seed + $j))
      ${train_cmd} ${odir}/log/train.${e}.${j}.log \
        ${train_script} --job ${j} --seed ${job_seed} ${gpu_opts} ${init_opts} ${resume_opts} || touch ${odir}/.error &
      sleep 10
    done
    wait
  )
  
  [ -f ${odir}/.error ] && echo "$0: error on iteration ${e} of training" && exit 1;
  
  # Model averaging
  combine_models=""
  for j in `seq 1 $nj`; do
    combine_models="${combine_models} ${odir}/${e}.${j}.mdl"
  done
  
  combine_models.py ${odir}/${e}.mdl ${odir}/conf.1.json --models ${combine_models} > ${odir}/log/combine.${e}.log
  resume_opts="--resume ${e}.mdl"
  init_opts=""
done


