#!/bin/bash

. ./path.sh

batchsize=512
skip_datadump=false
modelname=final.mdl
prior_scale=1.0
prior_floor=-20.0
prior_name="priors"
min_active=200
max_active=7000
max_mem=50000000
lattice_beam=8.0
beam=15.0
acoustic_scale=0.1
post_decode_acwt=10.0 # 10.0 for chain systems, 1.0 for non-chain
mean_var="(True, True)"

min_lmwt=6
max_lmwt=18
nj=80
stage=0

. ./utils/parse_options.sh
if [ $# -ne 4 ]; then
  echo "Usage: ./decode_nnet_pytorch.sh <data> <pytorch_model> <graphdir> <odir>"
  echo " --batchsize ${batchsize} --skip-datadump ${skip_datadump}"
  echo " --modelname ${modelname} --prior-scale ${prior_scale} --prior-floor ${prior_floor} --prior-name ${prior_name}"
  echo " --min-active ${min_active} --max-active ${max_active}"
  echo " --max-mem ${max_mem} --lattice-beam ${lattice_beam}"
  echo " --beam ${beam} --acoustic-scale ${acoustic_scale} --post-decode-acwt ${post_decode_acwt}"
  echo " --nj ${nj}"
  exit 1;
fi

data=$1
pytorch_model=$2
graphdir=$3
odir=$4

# We assume the acoustic model (trans.mdl) is 1 level above the graphdir
amdir=`dirname ${graphdir}`
trans_mdl=${amdir}/final.mdl
words_file=${graphdir}/words.txt
hclg=${graphdir}/HCLG.fst

skip_datadump_opts=
if $skip_datadump; then
  skip_datadump_opts="--skip-datadump"
else
  memmap_data.py ${data}/feats.scp ${data}/feats.scp.dat  
  skip_datadump_opts="--skip-datadump"
fi

mkdir -p ${odir}/log

decode_cmd="utils/queue.pl --mem 2G -l hostname=b0*" # The 'a' machines are just too slow
if [ $stage -le 0 ]; then
  segments=${data}/segments
  if [ ! -f ${data}/segments ]; then
    echo "No segments file found. Assuming wav.scp is indexed by utterance"
    segments=${data}/wav.scp
  fi

${decode_cmd} JOB=1:${nj} ${odir}/log/decode.JOB.log \
    ./utils/split_scp.pl -j ${nj} \$\[JOB -1\] ${segments} \|\
    decode.py ${skip_datadump_opts} \
      --datadir ${data} \
      --modeldir ${pytorch_model} \
      --dumpdir ${odir} \
      --modelname ${modelname} \
      --prior-scale ${prior_scale} \
      --prior-floor ${prior_floor} \
      --prior-name ${prior_name} \
      --words-file ${words_file} \
      --trans-mdl ${trans_mdl} \
      --hclg ${hclg} \
      --min-active ${min_active} \
      --max-active ${max_active} \
      --lattice-beam ${lattice_beam} \
      --beam ${beam} \
      --acoustic-scale ${acoustic_scale} \
      --post-decode-acwt ${post_decode_acwt} \
      --job JOB \
      --utt-subset /dev/stdin \
      --batchsize ${batchsize}
fi

if [ $stage -le 1 ]; then
  ./local/score.sh --cmd "$decode_cmd" \
    --min-lmwt ${min_lmwt} --max-lmwt ${max_lmwt} --word-ins-penalty 0.0 \
    ${data} ${graphdir} ${odir}
fi
