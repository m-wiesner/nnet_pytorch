#!/bin/bash

. ./path.sh

batchsize=512
checkpoint=final.mdl
cmd="utils/queue.pl --mem 6G -l hostname='!b02*&!b13*&!a*&!c06*&!c07*&!c12*&!c23*&!c24*&!c25*&!c26*&!c27*'" 
chunk_width=
output_idx=0

nj=80
stage=0

. ./utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: ./extract_embeddings.sh <data> <pytorch_model> <odir>"
  echo " --batchsize ${batchsize} "
  echo " --nj ${nj}"
  exit 1;
fi

data=$1
pytorch_model=$2
odir=$3

# We assume the acoustic model (trans.mdl) is 1 level above the graphdir

mkdir -p ${odir}/log

if [ $stage -le 0 ]; then
  segments=${data}/segments
  if [ ! -f ${data}/segments ]; then
    echo "No segments file found. Assuming wav.scp is indexed by utterance"
    segments=${data}/wav.scp
  fi
  
  cw_opts=
  if [ ! -z $chunk_width ]; then
    cw_opts="--chunk-width ${chunk_width}"
  fi 

  ${cmd} JOB=1:${nj} ${odir}/log/extract.JOB.log \
    ./utils/split_scp.pl -j ${nj} \$\[JOB -1\] ${segments} \|\
    extract_embeddings.py --datadir ${data} \
      --modeldir ${pytorch_model} \
      --dumpdir ${odir} \
      --checkpoint ${checkpoint} \
      --utt-subset /dev/stdin \
      --output-idx ${output_idx} \
      ${cw_opts}
fi
