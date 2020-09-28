#!/bin/bash

. ./cmd.sh
. ./path.sh

stage=0
subsampling=4
chaindir=exp/chain_wrn
model_dirname=wrn_semisup
checkpoint=20.mdl
target="2697 2697 2697 2697 2697 2697 2697 2697 2697 2697 2697 2697 2697 2697 2697"
idim=80

set -euo pipefail

tree=${chaindir}/tree

# Generation
modeldir=`dirname ${chaindir}`/${model_dirname}
gen_dir=${modeldir}/generate_cond_${checkpoint}
mkdir -p ${gen_dir}
generate_cmd="./utils/queue.pl --mem 2G --gpu 1 --config conf/gpu.conf ${gen_dir}/log"
${generate_cmd} generate_conditional_from_buffer.py \
  --gpu \
  --target ${target} \
  --idim ${idim} \
  --modeldir ${modeldir} --modelname ${checkpoint} \
  --dumpdir ${gen_dir} --batchsize 32

exit 0;
