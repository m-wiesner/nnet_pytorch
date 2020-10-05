#!/bin/bash

data=/export/a15/vpanayotov/data
subsampling=4
num_split=20
testsets="dev-clean dev-other test-clean test-other"
feat_affix=_fbank
standard_split=true

. ./cmd.sh
. ./path.sh

. ./utils/parse_options.sh

set -euo pipefail

for part in $testsets; do
  echo "-------------- Making ${part} ----------------------"
  dataname=$(echo ${part} | sed s/-/_/g)
  if $standard_split; then
    local/data_prep.sh $data/LibriSpeech/${part} data/${dataname}
  else
    echo "Assuming the testset ${part} is manually created and exists ..."
  fi
  ./utils/copy_data_dir.sh data/${dataname} data/${dataname}${feat_affix}
  ./steps/make_fbank.sh --cmd "$train_cmd" --nj 32 \
    data/${dataname}${feat_affix} exp/make_fbank/${dataname}${feat_affix} ${feat_affix##_}
  ./utils/fix_data_dir.sh data/${dataname}${feat_affix}
  ./steps/compute_cmvn_stats.sh data/${dataname}${feat_affix}
  ./utils/fix_data_dir.sh data/${dataname}${feat_affix}

  prepare_unlabeled_tgt.py --subsample ${subsampling} \
    data/${dataname}${feat_affix}/utt2num_frames > data/${dataname}${feat_affix}/pdfid.${subsampling}.tgt
  split_memmap_data.sh data/${dataname}${feat_affix} data/${dataname}${feat_affix}/pdfid.${subsampling}.tgt $num_split 
done

exit 0;
 

