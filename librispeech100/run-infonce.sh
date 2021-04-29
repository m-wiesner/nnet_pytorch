#!/bin/bash

# This is based almost entirely on the Kaldi librispeech recipe
# Change this location to somewhere where you want to put the data.
# This recipe ASSUMES YOU HAVE DOWNLOADED the Librispeech data
unlabeled_data=/export/corpora5 #/PATH/TO/LIBRISPEECH/data

. ./cmd.sh
. ./path.sh

stage=1
subsampling=4
traindir=data/train_100h
feat_affix=_fbank_64
chaindir=exp/chain
model_dirname=wrn_semisup
width=10
depth=28
l2=0.0001
lr=0.0002
decay=1e-05
weight_decay=1e-07
warmup=15000
batches_per_epoch=250
num_epochs=1000
nj_init=2
nj_final=6
perturb="[('time_mask', {'width': 20, 'max_drop_percent': 2.0}), ('freq_mask', {'width': 20, 'holes': 10}), ('gauss', {'std': 1.0})]"
chunkwidth=276
min_chunkwidth=4
random_cw=True
left_context=10
right_context=5
batchsize=8
delay_updates=1
grad_thresh=3.0
seed=0
resume=
num_split=80 # number of splits for memory-mapped data for training
. ./utils/parse_options.sh

set -euo pipefail

tree=${chaindir}/tree
targets=${traindir}${feat_affix}/pdfid.${subsampling}.tgt
trainname=`basename ${traindir}`

# Make the unlabeled data
if [ $stage -le 0 ]; then
  for part in train-clean-360 train-other-500; do
    local/data_prep.sh $unlabeled_data/LibriSpeech/${part} data/$(echo ${part} | sed s/-/_/g)
  done 

  ./utils/combine_data.sh data/train_860 data/train_{clean_360,other_500}
  ./steps/make_fbank.sh --cmd "$train_cmd" --nj 32 data/train_860 exp/make_fbank/train_860 ${feat_affix##_}
  ./utils/fix_data_dir.sh data/train_860
  ./steps/compute_cmvn_stats.sh data/train_860
  ./utils/fix_data_dir.sh data/train_860

  python prepare_unlabeled_tgt.py --subsample ${subsampling} data/train_860/utt2num_frames > data/train_860/pdfid.${subsampling}.tgt
  split_memmap_data.sh data/train_860 data/train_860/pdfid.${subsampling}.tgt ${num_split} 
fi


# We use a lower learning rate in order to prevent the model from forgetting
# too much. 
if [ $stage -eq 1 ]; then
  resume_opts=
  if [ ! -z $resume ]; then
    resume_opts="--resume ${resume}"
  fi 
  num_pdfs=$(tree-info ${tree}/tree | grep 'num-pdfs' | cut -d' ' -f2)
  train_script="train.py ${resume_opts} \
    --gpu \
    --expdir `dirname ${chaindir}`/${model_dirname} \
    --objective InfoNCE \
    --num-targets ${num_pdfs} \
    --subsample ${subsampling} \
    --model ChainWideResnet \
    --depth ${depth} \
    --width ${width} \
    --warmup ${warmup} \
    --decay ${decay} \
    --weight-decay ${weight_decay} \
    --lr ${lr} \
    --optim adam \
    --l2 ${l2} \
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

  train_async_parallel.sh ${resume_opts} \
    --cmd "$train_cmd" \
    --nj-init ${nj_init} \
    --nj-final ${nj_final} \
    --num-epochs ${num_epochs} \
    --seed ${seed} \
    "${train_script}" `dirname ${chaindir}`/${model_dirname}

fi

