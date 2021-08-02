#!/bin/bash

# Copyright 2021 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

. ./path.sh
. ./cmd.sh

subsampling=4
feat_affix=_fbank_64

. ./utils/parse_options.sh

if [ $# -ne 1 ]; then
    echo "Usage: ./local/nnet_pytorch/prepare_test.sh <testdir>"
    exit 1;
fi

data=$1

./utils/copy_data_dir.sh ${data} ${data}${feat_affix}
./steps/make_fbank.sh --cmd "$train_cmd" --nj 32 ${data}${feat_affix}
./utils/fix_data_dir.sh ${data}${feat_affix}
./steps/compute_cmvn_stats.sh ${data}${feat_affix}
./utils/fix_data_dir.sh ${data}${feat_affix}

prepare_unlabeled_tgt.py --subsample ${subsampling} \
  ${data}${feat_affix}/utt2num_frames > ${data}${feat_affix}/pdfid.${subsampling}.tgt
split_memmap_data.sh ${data}${feat_affix} ${data}${feat_affix}/pdfid.${subsampling}.tgt 10
