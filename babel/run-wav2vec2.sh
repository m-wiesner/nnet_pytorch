#!/bin/bash

. ./cmd.sh
. ./path.sh

gpu_config=conf/gpu.conf
subsampling=640
traindir=data/multi_train
targets=data/multi_train_raw_norm/pdfid.4.3500.tgt
feat_affix=_raw_norm
chaindir=exp/chain_wrn_3500
model_dirname=wav2vec2_lfmmi
l2=0.0001
xent=0.1
lr=0.0001
leaky_hmm=0.1
decay=1e-04
weight_decay=1e-07
warmup=4000
batches_per_epoch=1000
num_epochs=10
nj_init=2
nj_final=12
perturb="[('time_mask', {'width': 1600, 'max_drop_percent': 0.5}), ('gauss', {'std': 0.01})]" # std 0.1 before
chunkwidth=35200
min_chunkwidth=9600
random_cw=True
cw_curriculum=0.0
left_context=1600
right_context=800
batchsize=24
delay_updates=1
grad_thresh=32.0
freeze_feat_extractor=true
fp16=true

wav2vec2_name="facebook/wav2vec2-large-xlsr-53"

seed=0
resume=
. ./utils/parse_options.sh

set -eo pipefail

tree=${chaindir}/tree

resume_opts=
if [ ! -z $resume ]; then
  resume_opts="--resume ${resume}"
fi

fp16_opts=
if $fp16; then
  fp16_opts="--fp16"
fi

freeze_opts=
if $freeze_feat_extractor; then
  freeze_opts="--wav2vec2-freeze-feat-extractor"
fi

num_pdfs=$(tree-info ${tree}/tree | grep 'num-pdfs' | cut -d' ' -f2)
train_script="train.py \
  --gpu ${fp16_opts} \
  --leaky-hmm 0.1 \
  --denom-graph ${chaindir}/den.fst \
  --expdir `dirname ${chaindir}`/${model_dirname} \
  --objective Multitask \
  --multitask-losses \"[('LFMMIOnly', 1.0, 0), ('CrossEntropy', 0.1, 1), ('L2', 0.0001, 0)]\" \
  --num-targets ${num_pdfs} \
  --subsample ${subsampling} \
  --model ChainWav2Vec2 \
  --wav2vec2-mdl-name ${wav2vec2_name} ${freeze_opts} \
  --wav2vec2-subsampling ${subsampling} \
  --warmup ${warmup} \
  --decay ${decay} \
  --weight-decay ${weight_decay} \
  --lr ${lr} \
  --min-lr 1e-09 \
  --optim adam \
  --batches-per-epoch ${batches_per_epoch} \
  --num-epochs 1 \
  --delay-updates 1 \
  --grad-thresh ${grad_thresh} \
  --datasets \"[ \
    {\
  'data': '${traindir}${feat_affix}', \
  'tgt': '${targets}', 'batchsize': ${batchsize}, \
  'chunk_width': ${chunkwidth}, 'min_chunk_width': 8000, \
  'left_context': ${left_context}, 'right_context': ${right_context}, \
  'num_repeats': 1, 'mean_norm': False, 'var_norm': False, \
  'random_cw': ${random_cw}, 'cw_curriculum': ${cw_curriculum}, \
  'objf_names': ['LFMMIOnly', 'CrossEntropy', 'L2'], \
  'perturb_type': '''${perturb}'''\
    },\
  ]\"\
  "

train_cmd="utils/retry.pl utils/queue.pl --mem 12G --gpu 1 --config ${gpu_config}"

train_async_parallel.sh ${resume_opts} ${init_opts} \
  --cmd "$train_cmd" \
  --nj-init ${nj_init} \
  --nj-final ${nj_final} \
  --num-epochs ${num_epochs} \
  --seed ${seed} \
  --keep-every 2 \
  "${train_script}" `dirname ${chaindir}`/${model_dirname}

