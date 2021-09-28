#!/bin/bash

# This is based almost entirely on the Kaldi librispeech recipe
# Change this location to somewhere where you want to put the data.
# This recipe ASSUMES YOU HAVE DOWNLOADED the Librispeech data

. ./cmd.sh
. ./path.sh

subsampling=4
traindir=data/multi_train_fbank_64
chaindir=exp/chain_wrn_35000
model_dirname=wrn_lfmmi_init_multi
width=10
depth=28
strides="[1, 1, 2, 2]"
l2=0.0001
xent=0.1
infonce=0.0
lr=0.0001
leaky_hmm=0.1
#decay=2e-03
decay=1e-05
#weight_decay=1e-04
weight_decay=1e-07
#warmup=100
warmup=15000
#batches_per_epoch=100
batches_per_epoch=500
num_epochs=50
nj_init=1
nj_final=4
perturb="[('time_mask', {'width': 10, 'max_drop_percent': 0.5}), ('freq_mask', {'width': 10, 'holes': 5}), ('gauss', {'std': 0.1})]"
#perturb="none"
chunkwidth=220
min_chunkwidth=60
random_cw=True
left_context=10
right_context=5
batchsize=24
delay_updates=1
grad_thresh=32.0
seed=0
fp16="--fp16"
resume=
init=../babel/exp/wrn_lfmmi_500_fp16/140_150.mdl
freeze=false
replace_output=true
targets=data/multi_train_fbank_64/pdfid.4.tgt

. ./utils/parse_options.sh

set -euo pipefail

tree=${chaindir}/tree
trainname=`basename ${traindir}`

resume_opts=
if [ ! -z $resume ]; then
  resume_opts="--resume ${resume}"
fi 

freeze_opts=""
if $freeze; then
  freeze_opts="--freeze"
fi

replace_output_opts=""
if $replace_output; then
  replace_output_opts="--replace-output"
fi

num_pdfs=$(tree-info ${tree}/tree | grep 'num-pdfs' | cut -d' ' -f2)
train_script="train.py ${resume_opts} ${freeze_opts} ${replace_output_opts} \
  --gpu ${fp16} \
  --expdir `dirname ${chaindir}`/${model_dirname} \
  --objective Multitask \
  --multitask-losses \"[('LFMMIOnly', 1.0, 0), ('CrossEntropy', ${xent}, 1), ('L2', ${l2}, 0)]\" \
  --denom-graph ${chaindir}/den.fst \
  --leaky-hmm ${leaky_hmm} \
  --num-targets ${num_pdfs} \
  --subsample ${subsampling} \
  --model ChainWideResnet \
  --depth ${depth} \
  --width ${width} \
  --strides \"${strides}\" \
  --warmup ${warmup} \
  --decay ${decay} \
  --weight-decay ${weight_decay} \
  --lr ${lr} \
  --optim adam \
  --batches-per-epoch ${batches_per_epoch} \
  --num-epochs 1 \
  --delay-updates ${delay_updates} \
  --grad-thresh ${grad_thresh} \
  --datasets \"[ \
      {\
   'data': '${traindir}', \
   'tgt': '${targets}', \
   'batchsize': ${batchsize}, \
   'chunk_width': ${chunkwidth}, 'min_chunk_width': ${min_chunkwidth}, \
   'left_context': ${left_context}, 'right_context': ${right_context}, 'num_repeats': 1, \
   'mean_norm': True, 'var_norm': False, 'random_cw': ${random_cw}, 'cw_curriculum': 0.0, \
   'objf_names': ['LFMMIOnly', 'CrossEntropy', 'L2'], \
   'perturb_type': '''${perturb}'''\
     },\
   ]\"\
   "

train_cmd="utils/retry.pl utils/queue.pl --mem 12G --gpu 1 --config conf/gpu.conf"

init_opts=""
if [ ! -z $init ]; then
  init_opts="--init ${init}"
fi

train_async_parallel.sh ${resume_opts} ${init_opts} \
  --cmd "$train_cmd" \
  --nj-init ${nj_init} \
  --nj-final ${nj_final} \
  --num-epochs ${num_epochs} \
  --seed ${seed} \
  "${train_script}" `dirname ${chaindir}`/${model_dirname}


