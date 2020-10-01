#!/bin/bash

. ./cmd.sh
. ./path.sh

stage=0
subsampling=4
traindir=data/train_960
feat_affix=_fbank
chaindir=exp/chain_wrn
num_leaves=7000
model_dirname=wrn
batches_per_epoch=500
num_epochs=300
train_nj=4
resume=
num_split=20 # number of splits for memory-mapped data for training
average=true

. ./utils/parse_options.sh

set -euo pipefail

tree=${chaindir}/tree
targets=${traindir}${feat_affix}/pdfid.${subsampling}.tgt
trainname=`basename ${traindir}`

if [ $stage -le 1 ]; then
  echo "Creating Chain Topology, Denominator Graph, and nnet Targets ..."
  lang=data/lang_chain
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo

  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor ${subsampling} \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" ${num_leaves} ${traindir} \
    $lang exp/tri5b_ali_${trainname} ${tree}

  ali-to-phones ${tree}/final.mdl ark:"gunzip -c ${tree}/ali.*.gz |" ark:- |\
    chain-est-phone-lm --num-extra-lm-states=2000 ark:- ${chaindir}/phone_lm.fst

  chain-make-den-fst ${tree}/tree ${tree}/final.mdl \
    ${chaindir}/phone_lm.fst ${chaindir}/den.fst ${chaindir}/normalization.fst

  ali-to-pdf ${tree}/final.mdl ark:"gunzip -c ${tree}/ali.*.gz |" ark,t:${targets}
fi

if [ $stage -le 2 ]; then
  echo "Dumping memory mapped features ..."
  split_memmap_data.sh ${traindir}${feat_affix} ${targets} ${num_split} 
fi

# Multigpu training of Chain-WideResNet with optimizer state averaging
if [ $stage -le 3 ]; then
  resume_opts=
  if [ ! -z $resume ]; then
    resume_opts="--resume ${resume}"
  fi 

  num_pdfs=$(tree-info ${tree}/tree | grep 'num-pdfs' | cut -d' ' -f2)
  ./local/train_async_parallel.sh ${resume_opts} \
    --gpu true \
    --objective LFMMI \
    --denom-graph ${chaindir}/den.fst \
    --num-pdfs ${num_pdfs} \
    --subsample ${subsampling} \
    --model ChainWideResnet \
    --depth 28 \
    --width 10 \
    --warmup 20000 \
    --decay 1e-05 \
    --xent 0.05 \
    --l2 0.0001 \
    --weight-decay 1e-08 \
    --lr 0.0002 \
    --batches-per-epoch 500 \
    --num-epochs 300 \
    --validation-spks 0 \
    --nj 4 \
    "[ \
        {\
    'data': '${traindir}${feat_affix}', \
    'tgt': '${targets}', \
    'batchsize': 32, 'chunk_width': 140, \
    'left_context': 10, 'right_context': 5, \
    'mean_norm': True, 'var_norm': 'norm'
        }\
     ]" \
    `dirname ${chaindir}`/${model_dirname}
fi

# Average the last 60 epochs
if $average; then
  echo "Averaging the last few epochs ..."
  average_models.py `dirname ${chaindir}`/${model_dirname} 80 240 300
fi

