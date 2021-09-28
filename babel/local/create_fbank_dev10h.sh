#!/bin/bash

. ./path.sh
. ./cmd.sh

. ./utils/parse_options.sh

langid=$(cat conf/test.list | head -1)

data=data/${langid}_dev10h

./local/prepare_babel_data.sh --skip-train true --make-dev true ${langid}
./utils/copy_data_dir.sh ${data} ${data}_fbank_64
./steps/make_fbank.sh --cmd "$train_cmd" --nj 100 ${data}_fbank_64 
./steps/compute_cmvn_stats.sh ${data}_fbank_64

prepare_unlabeled_tgt.py --subsample 4 ${data}_fbank_64/utt2num_frames \
  > ${data}_fbank_64/pdfid.4.tgt

split_memmap_data.sh --raw true \
  ${data}_raw_norm ${data}_raw_norm/pdfid.640.tgt 20 
