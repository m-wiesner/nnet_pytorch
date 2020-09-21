#!/bin/bash

. ./path.sh
. ./cmd

if [ $# -ne 1 ]; then
  echo "Usage: ./local/prepare_librilight.sh <data>"
  exit 1;
fi

data=$1
# Get librilight set
wget https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz
tar -xvf librispeech_finetuning.tgz && mv librispeech_finetuning ${data}

# The following are the data subsets:
# 1h/{0..5}/{clean,other}
# 9h/{clean,other}
#
# In each of these subsets there speaker directories named with a speaker-id.
# Inside each directory are more directories corresponding to a recording-id.
# Within each speaker-id/recording-id subdirectory are the .flac audio files
# corresponding to speech utterances, as well as a .trans.txt file that has
# the transcription.

find -L $data -name "*.flac"

for part in 1h/{0..5}/{clean,other} 9h/{clean,other}; do
  dataname=$(echo ${part} | sed 's/\//_/g')
  ./local/prepare_librilight_dataset.sh ${data}/${part} data/train_${dataname}
done

./utils/combine_data.sh \
  data/train_10h data/train_1h_{0..5}_{clean,other} data/train_9h_{clean,other}
