#!/bin/bash
. ./path.sh
. ./cmd.sh

stage=0
subsampling=4
chaindir=exp/chain_wrn
model_dirname=wrn_semisup
checkpoint=20.mdl
top_k=10
target="2697 2697 2697 2697 2697"
left=10
right=5
chunk_width=20
idim=80
gpu=false

. ./utils/parse_options.sh

tree=${chaindir}/tree

# Generation
modeldir=`dirname ${chaindir}`/${model_dirname}
gen_dir=${modeldir}/generate_cond_${checkpoint}
mkdir -p ${gen_dir}

gpu_opts=
if $gpu; then
  gpu_opts="--gpu"
  generate_cmd="./utils/queue.pl --mem 2G --gpu 1 --config conf/gpu.conf ${gen_dir}/log"
else
  generate_cmd="./utils/queue.pl --mem 2G ${gen_dir}/log"
fi

target_opts=
if [ ! -z "$target" ]; then
  echo "Target: ${target}"
  target_opts="--target ${target}"
  generate_cmd="./utils/queue.pl --mem 2G --gpu 1 --config conf/gpu.conf ${gen_dir}/log"
  gpu_opts="--gpu"
else
  gpu_opts=
fi

${generate_cmd} generate_conditional_from_buffer.py \
  ${gpu_opts} \
  ${target_opts} \
  --idim ${idim} \
  --modeldir ${modeldir} --modelname ${checkpoint} \
  --dumpdir ${gen_dir} --batchsize 32 \
  --left-context ${left} --right-context ${right} --chunk-width ${chunk_width} \
  --top-k ${top_k}
