#!/bin/bash

. ./path.sh
. ./cmd.sh

. ./utils/parse_options.sh

langid=$(cat conf/test.list | head -1)

data=data/${langid}_dev10h

./local/prepare_babel_data.sh --skip-train true --make-dev true ${langid}
./utils/copy_data_dir.sh ${data} ${data}_raw_norm
python local/upsample_wav_scp.py ${data}/wav.scp > ${data}_raw_norm/wav.scp

wav-to-duration scp:${data}_raw_norm/wav.scp ark,t:- 2>/dev/null |\
  awk '{print 16000 * $2}' > ${data}_raw_norm/utt2num_frames 
prepare_unlabeled_tgt.py --subsample 640 ${data}_raw_norm/utt2num_frames \
  > ${data}_raw_norm/pdfid.640.tgt 

split_memmap_data.sh --raw true \
  ${data}_raw_norm ${data}_raw_norm/pdfid.640.tgt 20
