#!/bin/bash

. ./path.sh
. ./cmd.sh

if [ $# -ne 2 ]; then
  echo "Usage: ./local/prepare_librilight.sh <data> <kaldi_data>"
  exit 1;
fi

data=$1
kaldi_data=$2

data=$(./utils/make_absolute.sh ${data})
mkdir -p $kaldi_data
files=( `find -L ${data}/${p} -name "*.flac"` )

for f in ${files[@]}; do
  fname=`basename $f`
  fname=${fname%%.flac}
  echo "${fname} flac -c -d -s ${f} |" 
done | sort > ${kaldi_data}/wav.scp

paste -d' ' <(awk '{print $1}' ${kaldi_data}/wav.scp) \
            <(awk '{print $1}' ${kaldi_data}/wav.scp | cut -d'-' -f1) \
            > ${kaldi_data}/utt2spk

./utils/utt2spk_to_spk2utt.pl ${kaldi_data}/utt2spk > ${kaldi_data}/spk2utt

cat `find -L ${data}/${p} -name "*.trans.txt"` | sort > ${kaldi_data}/text
exit 0;
