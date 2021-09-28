#!/bin/bash

speech_data=/export/corpora5 #/PATH/TO/LIBRISPEECH/data

. ./cmd.sh
. ./path.sh

stage=1
decode_stage=0
subsampling=4
chaindir=exp/chain_wrn_3500
model_dirname=wrn_lfmmi_3500_fp16
checkpoint=60.mdl
acwt=1.0
cw=-1
lattice_beam=8.0
max_active=7000
min_active=200
beam=15.0
testsets="201_dev10h"
feat_affix="_fbank_64"
decode_nj=80
rescore=false
graph=graph_307

. ./utils/parse_options.sh

tree=${chaindir}/tree
post_decode_acwt=`echo ${acwt} | awk '{print 10*$1}'`

cp ${tree}/final.mdl exp/${model_dirname}/
cw_opts=""
if [ ! -z $cw ]; then
  cw_opts="--chunk-width ${cw}"
fi

for ds in $testsets; do 
  decode_nnet_pytorch.sh ${cw_opts} --min-lmwt 6 \
                         --stage ${decode_stage} \
                         --min-active ${min_active} \
                         --max-active ${max_active} \
                         --lattice-beam ${lattice_beam} \
                         --beam ${beam} \
                         --max-lmwt 18 \
                         --cmd "$decode_cmd" \
                         --checkpoint ${checkpoint} \
                         --acoustic-scale ${acwt} \
                         --post-decode-acwt ${post_decode_acwt} \
                         --nj ${decode_nj} \
                         data/${ds}${feat_affix} exp/${model_dirname} \
                         ${tree}/${graph} exp/${model_dirname}/decode_${checkpoint}_${graph}_${acwt}_cw${cw}_${ds}
done

