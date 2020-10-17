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
batches_per_epoch=250
num_epochs=240
train_nj=2
resume=
num_split=80 # number of splits for memory-mapped data for training
. ./utils/parse_options.sh

if [ $# -ne 1 ]; then
  echo "Usage: ./run-semisup-wrn.sh <seed_model>"
  echo "      This script assumes you have trained a seed model first."
  echo "      Do ./run-wrn.sh for instance."
  exit 1;
fi

init=$1
set -euo pipefail

[ ! -f ${init} ] && echo "Expected ${init} to exist." && exit 1; 
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
  train_async_parallel.sh ${resume_opts} \
    --gpu true \
    --objective SemisupLFMMI \
    --denom-graph ${chaindir}/den.fst \
    --num-pdfs ${num_pdfs} \
    --subsample ${subsampling} \
    --model ChainWideResnet \
    --depth 28 \
    --width 10 \
    --warmup 0 \
    --decay 1e-05 \
    --xent 0.01 \
    --l2 0.0001 \
    --weight-decay 1e-07 \
    --lr 0.00001 \
    --batches-per-epoch ${batches_per_epoch} \
    --num-epochs ${num_epochs} \
    --validation-spks 0 \
    --sgld-thresh 0 \
    --sgld-reinit-p 1.0 \
    --sgld-buffer 32 \
    --sgld-stepsize 1.0 \
    --sgld-steps 2 \
    --sgld-max-steps 50 \
    --sgld-noise 0.001 \
    --sgld-decay 0.0 \
    --sgld-warmup 0 \
    --sgld-optim adam \
    --sgld-replay-correction 0.0 \
    --l2-energy 0.0001 \
    --sgld-weight-decay 1e-03 \
    --delay-updates 2 \
    --lfmmi-weight 0.1 \
    --ebm-weight 1.0 \
    --nj ${train_nj} \
    --init ${init} \
    "[ \
        {\
    'data': '${traindir}${feat_affix}', \
    'tgt': '${targets}', \
    'batchsize': 32, 'chunk_width': 140, \
    'left_context': 10, 'right_context': 5, \
    'mean_norm': True, 'var_norm': 'norm' \
        },\
        {\
     'data': 'data/train_860', \
     'tgt': 'data/train_860/pdfid.${subsampling}.tgt', \
     'batchsize': 32, 'chunk_width': 30, \
     'left_context': 10, 'right_context': 5, \
     'mean_norm': True, 'var_norm': 'norm' \
       },\
     ]" \
     `dirname ${chaindir}`/${model_dirname}
fi

