#!/bin/bash

# Copyright 2021 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

. ./cmd.sh
. ./path.sh

ali_affix="_norm_cleaned_sp"
feat_affix="_fbank_64"
stage=0

. ./utils/parse_options.sh
if [ $# -ne 3 ]; then
  echo "Usage: ./local/nnet_pytorch/prepare_train.sh <data> <lang> <src>"
  exit 1;
fi

data=$1
lang=$2
src=$3

# We speed perturb the data and get the new alignments 
if [ $stage -le 0 ]; then
  ./utils/data/perturb_data_dir_speed_3way.sh ${data} ${data}_sp
  ./steps/make_mfcc.sh --nj 40 --cmd "$train_cmd" ${data}_sp
  ./steps/compute_cmvn_stats.sh ${data}_sp
  ./utils/fix_data_dir.sh ${data}_sp
  ./steps/align_fmllr.sh --cd "$train_cmd" --nj 40 \
    ${data}_sp ${lang} ${src} ${src}_ali${affix}
fi

# We create new volume perturbed fbank features for training for the speed
# perturbed data directory
if [ $stage -le 1 ]; then
  ./utils/copy_data_dir.sh ${data}_sp ${data}_sp${feat_affix}
  ./utils/data/perturb_data_dir_volume.sh ${data}_sp${feat_affix}
  
  ./steps/make_fbank.sh --cmd "$train_cmd" -nj 32 ${data}_sp
  ./utils/fix_data_dir.sh ${data}_sp${feat_affix}
  ./steps/compute_cmvn_stats.sh ${data}_sp${feat_affix}
  ./utils/fix_data_dir.sh ${data}_sp_fbank_64
fi
