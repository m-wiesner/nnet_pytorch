#!/bin/bash

speech_data=/export/corpora5 #/PATH/TO/LIBRISPEECH/data

. ./cmd.sh
. ./path.sh

stage=0
subsampling=4
chaindir=exp/chain
model_dirname=model1
checkpoint=180_220.mdl
acwt=1.0
testsets="dev_clean dev_other test_clean test_other"
decode_nj=80

set -euo pipefail

tree=${chaindir}/tree
post_decode_acwt=`echo ${acwt} | awk '{print 10*$1}'`

# Echo Make graph if it does not exist
if [ ! -f ${tree}/graph_tgsmall/HCLG.fst ]; then 
  ./utils/mkgraph.sh --self-loop-scale 1.0 \
    data/lang_test_tgsmall ${tree} ${tree}/graph_tgsmall
fi

## Prepare the test sets if not already done
if [ ! -f data/dev_clean_fbank/mapped/feats.dat.1 ]; then
  ./local/prepare_test.sh --subsampling ${subsampling} --data ${speech_data} 
fi

for ds in $testsets; do 
  ./local/decode_nnet_pytorch.sh --min-lmwt 6 \
                         --max-lmwt 18 \
                         --checkpoint ${checkpoint} \
                         --acoustic-scale ${acwt} \
                         --post-decode-acwt ${post_decode_acwt} \
                         --nj ${decode_nj} \
                         data/${ds}_fbank exp/${model_dirname} \
                         ${tree}/graph_tgsmall exp/${model_dirname}/decode_${checkpoint}_graph_${acwt}_${ds}
  
  echo ${decode_nj} > exp/${model_dirname}/decode_${checkpoint}_graph_${acwt}_${ds}/num_jobs
  ./steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
    data/lang_test_{tgsmall,fglarge} \
    data/${ds}_fbank exp/${model_dirname}/decode_${checkpoint}_graph_${acwt}_${ds}{,_fglarge_rescored} 
done

