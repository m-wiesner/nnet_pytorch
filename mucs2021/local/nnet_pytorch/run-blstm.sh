#!/bin/bash

# Copyright 2021 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0


. ./cmd.sh
. ./path.sh

stage=0
subsampling=4
traindir=data/train_norm_cleaned_sp
feat_affix=_fbank_64
chaindir=exp/chain_cleaned_sp
num_leaves=3500
model_dirname=blstm_librispeech
hdim=1024
nlayers=6
prefinal=512
l2=0.0001
xent=0.1
lr=0.00015
leaky_hmm=0.1
decay=1e-05
weight_decay=1e-07
warmup=15000
batches_per_epoch=250
num_epochs=250
nj_init=2
nj_final=4
perturb="[('time_mask', {'width': 10, 'max_drop_percent': 1.0}), ('freq_mask', {'width': 15, 'holes': 5}), ('gauss', {'std': 0.5})]"
chunkwidth=220
min_chunkwidth=60
random_cw=True
left_context=10
right_context=5
batchsize=32
delay_updates=1
grad_thresh=32.0
perturb_volume=true
init=
seed=0
resume=
num_split=80 # number of splits for memory-mapped data for training
average=true
. ./utils/parse_options.sh

set -euo pipefail

tree=${chaindir}/tree
targets=${traindir}${feat_affix}/pdfid.${subsampling}${target_affix}.tgt

# Training
if [ $stage -le 1 ]; then
  resume_opts=
  if [ ! -z $resume ]; then
    resume_opts="--resume ${resume}"
  fi
  
  init_opts=
  if [ ! -z $init ]; then
    init_opts="--init ${init}"
  fi
   
  num_pdfs=$(tree-info ${tree}/tree | grep 'num-pdfs' | cut -d' ' -f2)
  idim=$(feat-to-dim scp:${traindir}${feat_affix}/feats.scp -)
  train_script="train.py ${resume_opts} \
    --gpu \
    --expdir `dirname ${chaindir}`/${model_dirname} \
    --objective LFMMI \
    --denom-graph ${chaindir}/den.fst \
    --leaky-hmm ${leaky_hmm} \
    --num-targets ${num_pdfs} \
    --subsample ${subsampling} \
    --model ChainBLSTM \
    --blstm-idim ${idim} \
    --blstm-hdim ${hdim} \
    --blstm-num-layers ${nlayers} \
    --blstm-prefinal-dim ${prefinal} \
    --warmup ${warmup} \
    --decay ${decay} \
    --weight-decay ${weight_decay} \
    --lr ${lr} \
    --optim adam \
    --l2-reg ${l2} \
    --xent-reg ${xent} \
    --infonce-reg 0.0 \
    --batches-per-epoch ${batches_per_epoch} \
    --num-epochs 1 \
    --delay-updates ${delay_updates} \
    --grad-thresh ${grad_thresh} \
    --datasets \"[ \
        {\
     'data': '${traindir}${feat_affix}', \
     'tgt': '${targets}', \
     'batchsize': ${batchsize}, \
     'chunk_width': ${chunkwidth}, 'min_chunk_width': ${min_chunkwidth}, \
     'left_context': ${left_context}, 'right_context': ${right_context}, 'num_repeats': 1, \
     'mean_norm': True, 'var_norm': False, 'random_cw': ${random_cw}, \
     'perturb_type': '''${perturb}'''\
       },\
     ]\"\
     "
 
  train_cmd="utils/retry.pl utils/queue.pl --mem 4G --gpu 1 --config conf/gpu.conf"

  train_async_parallel.sh ${resume_opts} ${init_opts}\
    --cmd "$train_cmd" \
    --nj-init ${nj_init} \
    --nj-final ${nj_final} \
    --keep-last 40 \
    --num-epochs ${num_epochs} \
    --seed ${seed} \
    "${train_script}" `dirname ${chaindir}`/${model_dirname}
fi

# Average the last 40 epochs
if [ $stage -le 1 ]; then
  if $average; then
    feat_dim=$(feat-to-dim scp:${traindir}/feats.scp -)
    start_avg=$((num_epochs - 40))
    average_models.py `dirname ${chaindir}`/${model_dirname} ${feat_dim} ${start_avg} ${num_epochs}
  fi
fi
