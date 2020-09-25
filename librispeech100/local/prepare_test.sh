#!/bin/bash

data=/export/a15/vpanayotov/data
subsampling=4
num_split=20

. ./cmd.sh
. ./path.sh

. ./utils/parse_options.sh

set -euo pipefail

for part in dev-clean dev-other test-clean test-other; do
  echo "-------------- Making ${part} ----------------------"
  dataname=$(echo ${part} | sed s/-/_/g)
  local/data_prep.sh $data/LibriSpeech/${part} data/${dataname}
  ./utils/copy_data_dir.sh data/${dataname} data/${dataname}_fbank
  ./steps/make_fbank.sh --cmd "$train_cmd" --nj 32 \
    data/${dataname}_fbank exp/make_fbank/${dataname} fbank
  ./utils/fix_data_dir.sh data/${dataname}_fbank
  ./steps/compute_cmvn_stats.sh data/${dataname}_fbank
  ./utils/fix_data_dir.sh data/${dataname}_fbank

  ./local/split_memmap_data.sh data/${dataname}_fbank $num_split 
  python local/prepare_unlabeled_tgt.py --subsample ${subsampling} \
    data/${dataname}_fbank/utt2num_frames > data/${dataname}_fbank/pdfid.${subsampling}.tgt
done

exit 0;
 

