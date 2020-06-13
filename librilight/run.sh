#!/bin/bash

# This is based almost entirely on the Kaldi minilibrispeech recipe
# Change this location to somewhere where you want to put the data.
# This recipe ASSUMES YOU HAVE DOWNLOADED the Librispeech data
unlabeled_data= #/PATH/TO/LIBRISPEECH/data
data=./corpus
librilight=librilight
data_url=www.openslr.org/resources/31
lm_url=www.openslr.org/resources/11

. ./cmd.sh
. ./path.sh

stage=0
subsampling=4
chaindir=exp/chain
modelnum=1
. ./utils/parse_options.sh

set -euo pipefail

tree=${chaindir}/tree
mkdir -p $data


if [ $stage -le 0 ]; then
  local/download_lm.sh $lm_url $data data/local/lm
fi

if [ $stage -le 1 ]; then
  # format the data as Kaldi data directories
  local/prepare_dict.sh --stage 3 --nj 30 --cmd "$train_cmd" \
    data/local/lm data/local/lm data/local/dict_nosp

  utils/prepare_lang.sh data/local/dict_nosp \
    "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

  local/format_lms.sh --src-dir data/lang_nosp data/local/lm
  # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
    data/lang_nosp data/lang_nosp_test_tglarge
fi

if [ $stage -le 2 ]; then
  # Get the librislight semisupervised subsets
  local/prepare_librilight.sh $librilight
  ./steps/make_mfcc.sh --cmd "$train_cmd" --nj 32 data/train_10h exp/make_mfcc/train_10h mfcc
  ./utils/fix_data_dir.sh data/train_10h
  ./steps/compute_cmvn_stats.sh data/train_10h
  ./utils/fix_data_dir.sh data/train_10h

  utils/subset_data_dir.sh --shortest data/train_10h 500 data/train_500short
fi

# train a monophone system
if [ $stage -le 3 ]; then 
  steps/train_mono.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
    data/train_500short data/lang_nosp exp/mono
  
  steps/align_si.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
    data/train_10h data/lang_nosp exp/mono exp/mono_ali_train_10h
fi

# train a first delta + delta-delta triphone system on all utterances
if [ $stage -le 4 ]; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train_10h data/lang_nosp exp/mono_ali_train_10h exp/tri1

  steps/align_si.sh --nj 5 --cmd "$train_cmd" \
    data/train_10h data/lang_nosp exp/tri1 exp/tri1_ali_train_10h
fi

# train an LDA+MLLT system.
if [ $stage -le 5 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    data/train_10h data/lang_nosp exp/tri1_ali_train_10h exp/tri2b

  # Align utts using the tri2b model
  steps/align_si.sh  --nj 5 --cmd "$train_cmd" --use-graphs true \
    data/train_10h data/lang_nosp exp/tri2b exp/tri2b_ali_train_10h
fi

# Train tri3b, which is LDA+MLLT+SAT
if [ $stage -le 6 ]; then
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
    data/train_10h data/lang_nosp exp/tri2b_ali_train_10h exp/tri3b
fi

# Now we compute the pronunciation and silence probabilities from training data,
# and re-create the lang directory.
if [ $stage -le 7 ]; then
  steps/get_prons.sh --cmd "$train_cmd" \
    data/train_10h data/lang_nosp exp/tri3b
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp \
    exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
    exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict

  utils/prepare_lang.sh data/local/dict \
    "<UNK>" data/local/lang_tmp data/lang

  local/format_lms.sh --src-dir data/lang data/local/lm

  utils/build_const_arpa_lm.sh \
    data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge

  steps/align_fmllr.sh --nj 5 --cmd "$train_cmd" \
    data/train_10h data/lang exp/tri3b exp/tri3b_ali_train_10h
fi

# Make the unlabeled data
if [ $stage -le 9 ]; then
  for part in train-clean-100 train-clean-360 train-other-500; do
    local/data_prep.sh $unlabeled_data/LibriSpeech/${part} data/$(echo ${part} | sed s/-/_/g)
  done 

  ./utils/combine_data.sh data/train_960 data/train_{clean_100,clean_360,other_500}
  ./steps/make_fbank.sh --cmd "$train_cmd" --nj 32 data/train_960 exp/make_fbank/train_960 fbank
  ./utils/fix_data_dir.sh data/train_960
  ./steps/compute_cmvn_stats.sh data/train_960
  ./utils/fix_data_dir.sh data/train_960
fi

if [ $stage -le 10 ]; then
  traindir=data/train_10h
  feat_affix=_fbank
  ./utils/copy_data_dir.sh ${traindir} ${traindir}${feat_affix}
  ./steps/make_fbank.sh --cmd "$train_cmd" --nj 32 ${traindir}${feat_affix}
  ./utils/fix_data_dir.sh ${traindir}${feat_affix}
  ./steps/compute_cmvn_stats.sh ${traindir}${feat_affix}
  ./utils/fix_data_dir.sh ${traindir}${feat_affix}
fi

if [ $stage -le 11 ]; then
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
    --cmd "$train_cmd" 3500 data/train_10h \
    $lang exp/tri3b_ali_train_10h ${tree}

  ali-to-phones ${tree}/final.mdl ark:"gunzip -c ${tree}/ali.*.gz |" ark:- |\
    chain-est-phone-lm --num-extra-lm-states=2000 ark:- ${chaindir}/phone_lm.fst

  chain-make-den-fst ${tree}/tree ${tree}/final.mdl \
    ${chaindir}/phone_lm.fst ${chaindir}/den.fst ${chaindir}/normalization.fst
fi

if [ $stage -le 12 ]; then
  memmap_data.py data/train_10h_fbank/feats.scp data/train_10h_fbank/feats.scp.dat
  ali-to-pdf ${tree}/final.mdl ark:"gunzip -c ${tree}/ali.*.gz |" ark,t:data/train_10h_fbank/pdfid.${subsampling}.tgt
  memmap_data.py data/train_960/feats.scp data/train_960/feats.scp.dat
  python local/prepare_unlabeled_tgt.py --subsample ${subsampling} data/train_960/utt2num_frames > data/train_960/pdfid.${subsampling}.tgt
fi

# Supervised ChainWideResnet (Only works with subsampling == 4)
if [ $stage -eq 13 ]; then
  num_pdfs=$(tree-info ${tree}/tree | grep 'num-pdfs' | cut -d' ' -f2)
  ./train_nnet_pytorch.sh \
    --gpu true \
    --skip-datadump true \
    --objective LFMMI \
    --denom-graph ${chaindir}/den.fst \
    --num-pdfs ${num_pdfs} \
    --subsample ${subsampling} \
    --model ChainWideResnet \
    --depth 28 \
    --width 10 \
    --warmup 1000 \
    --decay 1e-05 \
    --xent 0.2 \
    --l2 0.0001 \
    --weight-decay 1e-05 \
    --lr 0.0001 \
    --batches-per-epoch 250 \
    --num-epochs 160 \
    --validation-spks 0 \
    "[ \
        {\
    'data': 'data/train_10h_fbank', \
    'tgt': 'data/train_10h_fbank/pdfid.${subsampling}.tgt', \
    'batchsize': 32, 'chunk_width': 140, \
    'left_context': 10, 'right_context': 5
        }\
     ]" \
    `dirname ${chaindir}`/model${modelnum}
fi

# Semi-Supervised ChainWideResnet (Only works with subsampling == 4)
if [ $stage -eq 14 ]; then
  num_pdfs=$(tree-info ${tree}/tree | grep 'num-pdfs' | cut -d' ' -f2)
  ./train_nnet_pytorch.sh \
    --gpu true \
    --skip-datadump true \
    --objective SemisupLFMMI \
    --denom-graph ${chaindir}/den.fst \
    --num-pdfs ${num_pdfs} \
    --subsample ${subsampling} \
    --model ChainWideResnet \
    --depth 28 \
    --width 10 \
    --warmup 1000 \
    --decay 1e-05 \
    --xent 0.2 \
    --l2 0.0001 \
    --weight-decay 1e-05 \
    --lr 0.0001 \
    --batches-per-epoch 250 \
    --num-epochs 160 \
    --validation-spks 0 \
    --sgld-thresh 0 \
    --sgld-reinit-p 0.05 \
    --sgld-buffer 10000 \
    --sgld-stepsize 1.0 \
    --sgld-steps 4 \
    --sgld-noise 0.001 \
    --sgld-decay 0.0 \
    --sgld-warmup 0 \
    --sgld-optim accsgld \
    --sgld-replay-correction 0.5 \
    --l2-energy 0.0001 \
    --sgld-weight-decay 1e-10 \
    --delay-updates 2 \
    --lfmmi-weight 1.0 \
    --ebm-weight 1.0 \
    "[ \
        {\
    'data': 'data/train_10h_fbank', \
    'tgt': 'data/train_10h_fbank/pdfid.${subsampling}.tgt', \
    'batchsize': 32, 'chunk_width': 140, \
    'left_context': 10, 'right_context': 5 \
        },\
        {\
     'data': 'data/train_960', \
     'tgt': 'data/train_960/pdfid.${subsampling}.tgt', \
     'batchsize': 32, 'chunk_width': 20, \
     'left_context': 10, 'right_context': 5 \
       },\
     ]" \
     `dirname ${chaindir}`/model${modelnum}
fi

# DECODING
if [ $stage -eq 15 ]; then
  ./utils/mkgraph.sh --self-loop-scale 1.0 \
    data/lang_test_tgsmall ${tree} ${tree}/graph_tgsmall
  ./local/prepare_test.sh --subsampling ${subsampling} --data ${unlabeled_data} 
  
  average_models.py `dirname ${chaindir}`/model${modelnum} 80 60 160  
  for ds in dev-clean dev-other test-clean test-other; do 
    ./decode_nnet_pytorch.sh --min-lmwt 6 \
                           --max-lmwt 18 \
                           --skip-datadump true \
                           --modelname 60_160.mdl \
                           --acoustic-scale 1.0 \
                           --post-decode-acwt 10.0 \
                           data/${ds}_fbank exp/model${modelnum} \
                           ${tree}/graph_tgsmall exp/model${modelnum}/decode_60_160.mdl_graph_${ds}
    ./steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_test_tg{small,large} \
      data/${ds}_fbank exp/model${modelnum}/decode_60_160.mdl_graph_${ds}{,_tglarge_rescored} 
  done
fi

# Generation
if [ $stage -eq 16 ]; then
  modeldir=`dirname ${chaindir}`/model${modelnum}
  gen_dir=${modeldir}/generate_cond_160.mdl
  mkdir -p ${gen_dir}
  generate_cmd="./utils/queue.pl --mem 2G --gpu 1 --config conf/gpu.conf ${gen_Dir}/log"
  ${generate_cmd} generate_conditional_from_buffer.py \
    --gpu \
    --target 1084 1084 1084 1084 1084 \
    --idim 80 --chunk-width 20 --left-context 4 --right-context 4 \
    --modeldir ${modeldir} --modelname 160.mdl \
    --dumpdir ${gen_dir} --batchsize 32
fi
