#!/bin/bash

# Copyright 2021 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0


. ./cmd.sh
. ./path.sh

stage=1
subsampling=4
chaindir=exp/chain_cleaned_sp
model_dirname=blstm_librispeech
checkpoint=120_160.mdl
acwt=1.0
cw=220
testsets="test"
feat_affix="_fbank_64"
decode_nj=200
rescore=false
graphname=graph_pd_expanded+extra
output_idx=0
lang=data/lang_pd_expanded+extra

. ./utils/parse_options.sh

tree=${chaindir}/tree
post_decode_acwt=`echo ${acwt} | awk '{print 10*$1}'`

# Echo Make graph if it does not exist
if [ ! -f ${tree}/${graphname}/HCLG.fst ]; then 
  ./utils/mkgraph.sh --self-loop-scale 1.0 \
    ${lang} ${tree} ${tree}/${graphname}
fi

cw_opts=""
if [ ! -z $cw ]; then
  cw_opts="--chunk-width ${cw}"
fi

for ds in $testsets; do
  [ -f data/${ds}/convs_dup ] && cp data/${ds}/convs_{,no}dup data/${ds}${feat_affix}/ 
  decode_nnet_pytorch.sh ${cw_opts} --min-lmwt 6 \
                         --max-lmwt 18 \
                         --cmd "$decode_cmd" \
                         --checkpoint ${checkpoint} \
                         --output-idx ${output_idx} \
                         --acoustic-scale ${acwt} \
                         --post-decode-acwt ${post_decode_acwt} \
                         --nj ${decode_nj} \
                         data/${ds}${feat_affix} exp/${model_dirname} \
                         ${tree}/${graphname} exp/${model_dirname}/decode_${checkpoint}_${graphname}_${acwt}_cw${cw}_${ds}
done

