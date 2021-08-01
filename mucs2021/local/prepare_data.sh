#!/bin/bash

# Copyright 2021 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

if [ $# -ne 1 ]; then
  echo "Usage: ./local/prepare_data.sh <data>"
  exit 1;
fi

data=$1

mkdir data
cp -r ${data}/train/transcripts data/train
cp -r ${data}/test/transcripts data/test

for i in train test; do
  mv data/${i}/wav.scp data/${i}/wav.scp.bk
  awk -v var=${data}/${i} '{print $1, var"/"$2}' data/${i}/wav.scp.bk > data/${i}/wav.scp
  ./utils/fix_data_dir.sh data/${i}
done

./analysis/identify_seen_utts.pl data/train/text data/test/text |\
  ./analysis/identify_seen_convs_matthew.pl |\
  awk '($2<80){print $1}' > data/test/convs_nodup
./analysis/identify_seen_utts.pl data/train/text data/test/text |\
  ./analysis/identify_seen_convs_matthew.pl |\
  awk '($2>=80){print $1}' > data/test/convs_dup


exit 0;
