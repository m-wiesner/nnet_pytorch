#!/bin/bash
. ./path.sh
idim=64
chunk_width=100
left_context=10
right_context=5
batchsize=32
perturb="none"
num_steps=

. ./utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: ./decorrupt.sh <data> <model> <checkpoint>"
  exit 1;
fi

data=$1
model=$2
checkpoint=$3

odir=${model}/decorrupt_${checkpoint}
mkdir -p ${odir}

num_steps_opts=""
if [ ! -z $num_steps ]; then
  num_steps_opts="--num-steps ${num_steps}"
fi

train_cmd="utils/retry.pl utils/queue.pl --mem 2G --gpu 1 --config conf/gpu.conf"

${train_cmd} ${odir}/log decorrupt.py --gpu \
  --datadir ${data} \
  --modeldir ${model} \
  --checkpoint ${checkpoint} \
  --dumpdir ${odir} \
  --idim ${idim} \
  --chunk-width ${chunk_width} \
  --left-context ${left_context} \
  --right-context ${right_context} \
  --batchsize ${batchsize} \
  --perturb ${perturb} \
  ${num_steps_opts}
