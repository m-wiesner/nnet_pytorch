#!/bin/bash

# Copyright 2021 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0


. ./path.sh
. ./cmd.sh

num_leaves=3500
lang_affix=_chain
subsampling=4
num_split=20

. ./utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: ./local/make_chain.sh --num-leaves n <data> <lang> <ali> <chaindir>"
  exit 1;
fi

data=$1
lang=$2
alidir=$3
odir=$4

echo "Creating Chain Topology, Denominator Graph, and nnet Targets ..."
lang_chain=${lang}${lang_affix}_${num_leaves}
cp -r $lang $lang_chain
silphonelist=$(cat $lang_chain/phones/silence.csl) || exit 1;
nonsilphonelist=$(cat $lang_chain/phones/nonsilence.csl) || exit 1;
targets=${data}/pdfid.${subsampling}.tgt

# Use our special topology... note that later on may have to tune this
# topology.
steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist > ${lang_chain}/topo

steps/nnet3/chain/build_tree.sh \
  --frame-subsampling-factor ${subsampling} \
  --context-opts "--context-width=2 --central-position=1" \
  --cmd "$train_cmd" ${num_leaves} ${data} \
  $lang_chain $alidir ${odir}/tree

ali-to-phones ${odir}/tree/final.mdl ark:"gunzip -c ${odir}/tree/ali.*.gz |" ark:- |\
  chain-est-phone-lm --num-extra-lm-states=2000 ark:- ${odir}/phone_lm.fst

chain-make-den-fst ${odir}/tree/tree ${odir}/tree/final.mdl \
  ${odir}/phone_lm.fst ${odir}/den.fst ${odir}/normalization.fst

ali-to-pdf ${odir}/tree/final.mdl ark:"gunzip -c ${odir}/tree/ali.*.gz |" ark,t:${targets}

num_spk=$(cat ${data}/spk2utt | wc -l)
if [ $num_spk -lt $num_split ]; then
  echo "Reducing num_split to ${num_spk}=num_spk as num_spk < $num_split"
  num_split=${num_spk}
fi
split_memmap_data.sh ${data} ${data}/pdfid.${subsampling}.tgt ${num_split}


