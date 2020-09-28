#!/bin/bash

. ./cmd.sh
. ./path.sh

stage=0
subsampling=3
chaindir=exp/chain_tdnn
model_dirname=tdnn
batches_per_epoch=250
num_epochs=240
train_nj=2
acwt=1.0
resume=
num_split=20 # number of splits for memory-mapped data for training
average=true

. ./utils/parse_options.sh

set -euo pipefail

tree=${chaindir}/tree
post_decode_acwt=`echo ${acwt} | awk '{print 10*$1}'`


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
    --cmd "$train_cmd" 3500 data/train_100h \
    $lang exp/tri3_ali_train_100h ${tree}

  ali-to-phones ${tree}/final.mdl ark:"gunzip -c ${tree}/ali.*.gz |" ark:- |\
    chain-est-phone-lm --num-extra-lm-states=2000 ark:- ${chaindir}/phone_lm.fst

  chain-make-den-fst ${tree}/tree ${tree}/final.mdl \
    ${chaindir}/phone_lm.fst ${chaindir}/den.fst ${chaindir}/normalization.fst

  ali-to-pdf ${tree}/final.mdl ark:"gunzip -c ${tree}/ali.*.gz |" ark,t:data/train_100h_fbank/pdfid.${subsampling}.tgt
fi


if [ $stage -le 2 ]; then
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
    --model ChainTDNN \
    --hdim 1024  \
    --num-layers 13 \
    --dropout 0.2 \
    --prefinal-dim 192 \
    --warmup 15000 \
    --decay 1e-05 \
    --xent 0.1 \
    --l2 0.0001 \
    --weight-decay 1e-07 \
    --lr 0.0002 \
    --batches-per-epoch ${batches_per_epoch} \
    --num-epochs ${num_epochs} \
    --validation-spks 0 \
    --nj ${train_nj} \
    "[ \
        {\
    'data': 'data/train_100h_fbank', \
    'tgt': 'data/train_100h_fbank/pdfid.${subsampling}.tgt', \
    'batchsize': 128, 'chunk_width': 140, \
    'left_context': 10, 'right_context': 5
        }\
     ]" \
    `dirname ${chaindir}`/${model_dirname}
fi

# Average the last 40 epochs
if $average; then
  echo "Averaging the last few epochs ..."
  average_models.py `dirname ${chaindir}`/${model_dirname} 80 200 240
fi
