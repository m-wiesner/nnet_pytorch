#!/bin/bash

# This is based almost entirely on the Kaldi librispeech recipe
# Change this location to somewhere where you want to put the data.
# This recipe ASSUMES YOU HAVE DOWNLOADED the Librispeech data
unlabeled_data=/export/corpora5 #/PATH/TO/LIBRISPEECH/data

. ./cmd.sh
. ./path.sh

stage=0
subsampling=4
chaindir=exp/chain_wrn
model_dirname=wrn_semisup
batches_per_epoch=100
num_epochs=240
train_nj=2
resume=
num_split=20 # number of splits for memory-mapped data for training
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

# Make the unlabeled data
if [ $stage -le 0 ]; then
  for part in train-clean-360 train-other-500; do
    local/data_prep.sh $unlabeled_data/LibriSpeech/${part} data/$(echo ${part} | sed s/-/_/g)
  done 

  ./utils/combine_data.sh data/train_860 data/train_{clean_360,other_500}
  ./steps/make_fbank.sh --cmd "$train_cmd" --nj 32 data/train_860 exp/make_fbank/train_860 fbank
  ./utils/fix_data_dir.sh data/train_860
  ./steps/compute_cmvn_stats.sh data/train_860
  ./utils/fix_data_dir.sh data/train_860

  split_memmap_data.sh data/train_860 ${num_split} 
  python prepare_unlabeled_tgt.py --subsample ${subsampling} data/train_860/utt2num_frames > data/train_860/pdfid.${subsampling}.tgt
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
    --warmup 500 \
    --decay 1e-05 \
    --xent 0.1 \
    --l2 0.0001 \
    --weight-decay 1e-07 \
    --lr 0.00001 \
    --batches-per-epoch ${batches_per_epoch} \
    --num-epochs ${num_epochs} \
    --validation-spks 0 \
    --sgld-thresh 0 \
    --sgld-reinit-p 0.05 \
    --sgld-buffer 10000 \
    --sgld-stepsize 1.0 \
    --sgld-steps 4 \
    --sgld-noise 0.001 \
    --sgld-decay 0.0 \
    --sgld-warmup 500 \
    --sgld-optim accsgld \
    --sgld-replay-correction 0.5 \
    --l2-energy 0.0001 \
    --sgld-weight-decay 1e-10 \
    --delay-updates 2 \
    --lfmmi-weight 1.0 \
    --ebm-weight 1.0 \
    --nj ${train_nj} \
    --init ${init} \
    "[ \
        {\
    'data': 'data/train_100h_fbank', \
    'tgt': 'data/train_100h_fbank/pdfid.${subsampling}.tgt', \
    'batchsize': 32, 'chunk_width': 140, \
    'left_context': 10, 'right_context': 5 \
        },\
        {\
     'data': 'data/train_860', \
     'tgt': 'data/train_860/pdfid.${subsampling}.tgt', \
     'batchsize': 16, 'chunk_width': 60, \
     'left_context': 10, 'right_context': 5 \
       },\
     ]" \
     `dirname ${chaindir}`/${model_dirname}
fi

