#!/bin/bash

# This is based almost entirely on the Kaldi librispeech recipe
# Change this location to somewhere where you want to put the data.
# This recipe ASSUMES YOU HAVE DOWNLOADED the Librispeech data
unlabeled_data=/export/corpora5 #/PATH/TO/LIBRISPEECH/data

. ./cmd.sh
. ./path.sh

stage=1
subsampling=4
traindir=data/train_100h
unsupdir=data/train_860h
feat_affix=_fbank_64
chaindir=exp/chain
model_dirname=wrn_semisup
batches_per_epoch=250
num_epochs=240
delay=2
train_nj_init=2
train_nj_final=6
ebm_weight=1.0
ebm_type="uncond"
ebm_tgt=data/train_100h_fbank_64/pdfid.4.tgt
sgld_opt=adam
sgld_stepsize=1.0
sgld_maxsteps=50.0
sgld_minsteps=1
sgld_replay=1.0
sgld_noise=0.001
sgld_weight_decay=1e-10
sgld_decay=1e-04
sgld_warmup=15000
sgld_clip=1.0
sgld_init_val=1.5
sgld_epsilon=1e-04
lr=0.0002
xent=0.1
l2=0.0001
leaky_hmm=0.1
l2_energy=0.001
warmup=15000
unsup_num_repeats=1
unsup_batchsize=32
unsup_chunkwidth=50
unsup_left=10
unsup_right=5
mean_norm=True
var_norm=True
perturb="gauss 0.01"
depth=28
width=10
seed=0
resume=
num_split=80 # number of splits for memory-mapped data for training
. ./utils/parse_options.sh

set -euo pipefail

tree=${chaindir}/tree
targets=${traindir}${feat_affix}/pdfid.${subsampling}.tgt
trainname=`basename ${traindir}`

# Make the unlabeled data
if [ $stage -le 0 ]; then
  for part in train-clean-360 train-other-500; do
    local/data_prep.sh $unlabeled_data/LibriSpeech/${part} data/$(echo ${part} | sed s/-/_/g)
  done 

  ./utils/combine_data.sh data/train_860 data/train_{clean_360,other_500}
  ./steps/make_fbank.sh --cmd "$train_cmd" --nj 32 data/train_860 exp/make_fbank/train_860 ${feat_affix##_}
  ./utils/fix_data_dir.sh data/train_860
  ./steps/compute_cmvn_stats.sh data/train_860
  ./utils/fix_data_dir.sh data/train_860

  python prepare_unlabeled_tgt.py --subsample ${subsampling} data/train_860/utt2num_frames > data/train_860/pdfid.${subsampling}.unsup.tgt
  split_memmap_data.sh data/train_860 data/train_860/pdfid.${subsampling}.tgt ${num_split} 
fi


# We use a lower learning rate in order to prevent the model from forgetting
# too much. 
if [ $stage -eq 1 ]; then
  resume_opts=
  if [ ! -z $resume ]; then
    resume_opts="--resume ${resume}"
  fi 
  num_pdfs=$(tree-info ${tree}/tree | grep 'num-pdfs' | cut -d' ' -f2)
  idim=$(feat-to-dim scp:${traindir}${feat_affix}/feats.scp -)
  train_async_parallel.sh ${resume_opts} \
    --gpu true \
    --objective SemisupLFMMI \
    --denom-graph ${chaindir}/den.fst \
    --num-pdfs ${num_pdfs} \
    --idim ${idim} \
    --subsample ${subsampling} \
    --model ChainWideResnet \
    --depth ${depth} \
    --width ${width} \
    --warmup ${warmup} \
    --decay 1e-05 \
    --xent ${xent} \
    --l2 ${l2} \
    --leaky-hmm ${leaky_hmm} \
    --weight-decay 1e-07 \
    --lr ${lr} \
    --batches-per-epoch ${batches_per_epoch} \
    --num-epochs ${num_epochs} \
    --validation-spks 0 \
    --sgld-thresh 0.0 \
    --sgld-reinit-p 0.05 \
    --sgld-buffer 10000 \
    --sgld-stepsize ${sgld_stepsize} \
    --sgld-steps ${sgld_minsteps} \
    --sgld-max-steps ${sgld_maxsteps} \
    --sgld-noise ${sgld_noise} \
    --sgld-decay ${sgld_decay} \
    --sgld-real-decay 0.0 \
    --sgld-clip ${sgld_clip} \
    --sgld-warmup ${sgld_warmup} \
    --sgld-optim ${sgld_opt} \
    --sgld-init-val ${sgld_init_val} \
    --sgld-epsilon ${sgld_epsilon} \
    --sgld-replay-correction ${sgld_replay} \
    --l2-energy ${l2_energy} \
    --sgld-weight-decay ${sgld_weight_decay} \
    --delay-updates ${delay} \
    --lfmmi-weight 1.0 \
    --ebm-weight ${ebm_weight} \
    --ebm-type ${ebm_type} \
    --ebm-tgt ${ebm_tgt} \
    --nj-init ${train_nj_init} \
    --nj-final ${train_nj_final} \
    --seed ${seed} \
    "[ \
        {\
    'data': '${traindir}${feat_affix}', \
    'tgt': '${targets}', \
    'batchsize': 32, 'num_repeats': 1, 'chunk_width': 140, \
    'left_context': 10, 'right_context': 5, \
    'mean_norm': ${mean_norm}, 'var_norm': ${var_norm}, 'perturb_type': '${perturb}' \
        },\
        {\
     'data': '${unsupdir}', \
     'tgt': '${unsupdir}/pdfid.${subsampling}.unsup.tgt', \
     'batchsize': ${unsup_batchsize}, 'num_repeats': ${unsup_num_repeats}, 'chunk_width': ${unsup_chunkwidth}, \
     'left_context': ${unsup_left}, 'right_context': ${unsup_right}, \
     'mean_norm': ${mean_norm}, 'var_norm': ${var_norm}, 'perturb_type': '${perturb}' \
       },\
     ]" \
     `dirname ${chaindir}`/${model_dirname}
fi

