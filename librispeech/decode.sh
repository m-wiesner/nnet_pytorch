#!/bin/bash

# This is based almost entirely on the Kaldi Librispeech recipe
# Change this location to somewhere where you want to put the data.
# This recipe ASSUMES YOU HAVE DOWNLOADED the Librispeech data
corpus_dir=/export/corpora5/LibriSpeech #/PATH/TO/LIBRISPEECH/data
mfccdir=mfcc

. ./cmd.sh
. ./path.sh

stage=0
subsampling=3
chaindir=exp/chain_sub${subsampling}
model_dirname=model_blstm
resume=
checkpoint=final.mdl
testsets="dev_clean dev_other test_clean test_other"
nj=40
decode_nj=40
num_split=20
. ./utils/parse_options.sh

set -euo pipefail

tree=${chaindir}/tree

# DECODING
if [ $stage -le 1 ]; then
  # Echo Make graph if it does not exist
  if [ ! -f ${tree}/graph_tgsmall/HCLG.fst ]; then 
    ./utils/mkgraph.sh --self-loop-scale 1.0 \
      data/lang_test_tgsmall ${tree} ${tree}/graph_tgsmall
  fi

  if ! [ -d data/dev_clean_fbank ]; then
    # Prepare the test sets
    for part in dev-clean test-clean dev-other test-other; do
       echo "-------------- Making ${part} ----------------------"
      # use underscore-separated names in data directories.
      local/data_prep.sh $corpus_dir/$part data/$(echo $part | sed s/-/_/g)_fbank
      dataname=$(echo ${part} | sed s/-/_/g)
      steps/make_fbank.sh --cmd "$train_cmd" --nj $nj \
        data/${dataname}_fbank exp/make_fbank/${dataname} fbank
      utils/fix_data_dir.sh data/${dataname}_fbank
      steps/compute_cmvn_stats.sh data/${dataname}_fbank
      utils/fix_data_dir.sh data/${dataname}_fbank

      ./local/split_memmap_data.sh data/${dataname}_fbank $num_split
      python local/prepare_unlabeled_tgt.py --subsample ${subsampling} \
        data/${dataname}_fbank/utt2num_frames > data/${dataname}_fbank/pdfid.${subsampling}.tgt
    done
  fi
fi

if [ $stage -le 2 ]; then
  # Average models (This gives better performance)
  #average_models.py `dirname ${chaindir}`/${model_dirname} 80 180 220 
  for ds in $testsets; do 
    ./decode_nnet_pytorch.sh --min-lmwt 6 \
                           --max-lmwt 18 \
                           --checkpoint ${checkpoint} \
                           --acoustic-scale 1.0 \
                           --post-decode-acwt 10.0 \
                           --nj ${decode_nj} \
                           data/${ds}_fbank exp/${model_dirname} \
                           ${tree}/graph_tgsmall exp/${model_dirname}/decode_${checkpoint}_graph_${ds}
    
    echo ${decode_nj} > exp/${model_dirname}/decode_${checkpoint}_graph_${ds}/num_jobs
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_test_{tgsmall,fglarge} \
      data/${ds}_fbank exp/${model_dirname}/decode_${checkpoint}_graph_${ds}{,_fglarge_rescored} 

    ./local/score.sh --cmd "$decode_cmd" \
      --min-lmwt 6 --max-lmwt 18 --word-ins-penalty 0.0 \
      data/${ds}_fbank ${tree}/graph_tgsmall exp/${model_dirname}/decode_${checkpoint}_graph_${ds}_fglarge_rescored
  done
fi

