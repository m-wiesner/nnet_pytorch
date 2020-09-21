#!/bin/bash

# This is based almost entirely on the Kaldi minilibrispeech recipe
# Change this location to somewhere where you want to put the data.
# This recipe ASSUMES YOU HAVE DOWNLOADED the Librispeech data
unlabeled_data=/export/corpora5 #/PATH/TO/LIBRISPEECH/data
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
modelname=60_160.mdl
testsets="dev_clean dev_other test_clean test_other"
decode_nj=80
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
  # Get the train-100 subset 
  #local/data_prep.sh ${unlabeled_data}/LibriSpeech/train-clean-100 data/train_100h
  #./steps/make_mfcc.sh --cmd "$train_cmd" --nj 32 data/train_100h exp/make_mfcc/train_100h mfcc
  ./utils/fix_data_dir.sh data/train_100h
  ./steps/compute_cmvn_stats.sh data/train_100h
  ./utils/fix_data_dir.sh data/train_100h

  utils/subset_data_dir.sh --shortest data/train_100h 500 data/train_500short
  utils/subset_data_dir.sh data/train_100h 5000 data/train_5k
  utils/subset_data_dir.sh data/train_100h 10000 data/train_10k
fi

# train a monophone system
if [ $stage -le 3 ]; then 
  steps/train_mono.sh --boost-silence 1.25 --nj 15 --cmd "$train_cmd" \
    data/train_500short data/lang_nosp exp/mono
  
  steps/align_si.sh --boost-silence 1.25 --nj 15 --cmd "$train_cmd" \
    data/train_5k data/lang_nosp exp/mono exp/mono_ali_train_5k
fi

# train a first delta + delta-delta triphone system on 5k utterances
if [ $stage -le 4 ]; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train_5k data/lang_nosp exp/mono_ali_train_5k exp/tri1

  steps/align_si.sh --nj 15 --cmd "$train_cmd" \
    data/train_10k data/lang_nosp exp/tri1 exp/tri1_ali_train_10k
fi

# train a first delta + delta-delta triphone system on 10k utterances
if [ $stage -le 5 ]; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2500 15000 data/train_10k data/lang_nosp exp/tri1_ali_train_10k exp/tri1b

  steps/align_si.sh --nj 20 --cmd "$train_cmd" \
    data/train_100h data/lang_nosp exp/tri1b exp/tri1b_ali_train_100h
fi

# train an LDA+MLLT system.
if [ $stage -le 6 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 4200 40000 \
    data/train_100h data/lang_nosp exp/tri1b_ali_train_100h exp/tri2

  # Align utts using the tri2b model
  steps/align_si.sh --nj 20 --cmd "$train_cmd" --use-graphs true \
    data/train_100h data/lang_nosp exp/tri2 exp/tri2_ali_train_100h
fi

# Train tri3, which is LDA+MLLT+SAT
if [ $stage -le 7 ]; then
  steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
    data/train_100h data/lang_nosp exp/tri2_ali_train_100h exp/tri3
fi

# Now we compute the pronunciation and silence probabilities from training data,
# and re-create the lang directory.
if [ $stage -le 8 ]; then
  steps/get_prons.sh --cmd "$train_cmd" \
    data/train_100h data/lang_nosp exp/tri3
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp \
    exp/tri3/pron_counts_nowb.txt exp/tri3/sil_counts_nowb.txt \
    exp/tri3/pron_bigram_counts_nowb.txt data/local/dict

  utils/prepare_lang.sh data/local/dict \
    "<UNK>" data/local/lang_tmp data/lang

  local/format_lms.sh --src-dir data/lang data/local/lm

  utils/build_const_arpa_lm.sh \
    data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge

  steps/align_fmllr.sh --nj 5 --cmd "$train_cmd" \
    data/train_100h data/lang exp/tri3 exp/tri3_ali_train_100h
fi

# Make the unlabeled data
if [ $stage -le 9 ]; then
  for part in train-clean-360 train-other-500; do
    local/data_prep.sh $unlabeled_data/LibriSpeech/${part} data/$(echo ${part} | sed s/-/_/g)
  done 

  ./utils/combine_data.sh data/train_860 data/train_{clean_360,other_500}
  ./steps/make_fbank.sh --cmd "$train_cmd" --nj 32 data/train_860 exp/make_fbank/train_860 fbank
  ./utils/fix_data_dir.sh data/train_860
  ./steps/compute_cmvn_stats.sh data/train_860
  ./utils/fix_data_dir.sh data/train_860
fi

if [ $stage -le 10 ]; then
  traindir=data/train_100h
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
    --cmd "$train_cmd" 3500 data/train_100h \
    $lang exp/tri3_ali_train_100h ${tree}

  ali-to-phones ${tree}/final.mdl ark:"gunzip -c ${tree}/ali.*.gz |" ark:- |\
    chain-est-phone-lm --num-extra-lm-states=2000 ark:- ${chaindir}/phone_lm.fst

  chain-make-den-fst ${tree}/tree ${tree}/final.mdl \
    ${chaindir}/phone_lm.fst ${chaindir}/den.fst ${chaindir}/normalization.fst
fi

if [ $stage -le 12 ]; then
  memmap_data.py data/train_100h_fbank/feats.scp data/train_100h_fbank/feats.scp.dat
  ali-to-pdf ${tree}/final.mdl ark:"gunzip -c ${tree}/ali.*.gz |" ark,t:data/train_100h_fbank/pdfid.${subsampling}.tgt
  memmap_data.py data/train_860/feats.scp data/train_860/feats.scp.dat
  python local/prepare_unlabeled_tgt.py --subsample ${subsampling} data/train_860/utt2num_frames > data/train_860/pdfid.${subsampling}.tgt
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
    --warmup 20000 \
    --decay 1e-05 \
    --xent 0.0 \
    --l2 0.0005 \
    --weight-decay 1e-06 \
    --lr 0.0002 \
    --batches-per-epoch 250 \
    --num-epochs 200 \
    --validation-spks 0 \
    --resume 360.mdl \
    "[ \
        {\
    'data': 'data/train_100h_fbank', \
    'tgt': 'data/train_100h_fbank/pdfid.${subsampling}.tgt', \
    'batchsize': 32, 'chunk_width': 140, \
    'left_context': 10, 'right_context': 5
        }\
     ]" \
    `dirname ${chaindir}`/model${modelnum}
fi

if [ $stage -eq 20 ]; then
  num_pdfs=$(tree-info ${tree}/tree | grep 'num-pdfs' | cut -d' ' -f2)
  ./local/train_async_parallel.sh \
    --gpu true \
    --skip-datadump true \
    --objective LFMMI \
    --denom-graph ${chaindir}/den.fst \
    --num-pdfs ${num_pdfs} \
    --subsample ${subsampling} \
    --model ChainWideResnet \
    --depth 28 \
    --width 10 \
    --warmup 15000 \
    --decay 1e-05 \
    --xent 0.1 \
    --l2 0.0001 \
    --weight-decay 1e-07 \
    --lr 0.0002 \
    --batches-per-epoch 250 \
    --num-epochs 200 \
    --validation-spks 0 \
    --nj 2 \
    --resume 78 \
    "[ \
        {\
    'data': 'data/train_100h_fbank', \
    'tgt': 'data/train_100h_fbank/pdfid.${subsampling}.tgt', \
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
    --warmup 15000 \
    --decay 1e-05 \
    --xent 0.1 \
    --l2 0.0001 \
    --weight-decay 1e-07 \
    --lr 0.0003 \
    --batches-per-epoch 250 \
    --num-epochs 250 \
    --validation-spks 0 \
    --sgld-thresh 0 \
    --sgld-reinit-p 0.05 \
    --sgld-buffer 10000 \
    --sgld-stepsize 1.0 \
    --sgld-steps 2 \
    --sgld-noise 0.0001 \
    --sgld-decay 0.0 \
    --sgld-warmup 0 \
    --sgld-optim accsgld \
    --sgld-replay-correction 1.0 \
    --l2-energy 0.0001 \
    --sgld-weight-decay 0.0 \
    --delay-updates 2 \
    --lfmmi-weight 1.0 \
    --ebm-weight 1.0 \
    "[ \
        {\
    'data': 'data/train_100h_fbank', \
    'tgt': 'data/train_100h_fbank/pdfid.${subsampling}.tgt', \
    'batchsize': 32, 'chunk_width': 140, \
    'left_context': 10, 'right_context': 5 \
        },\
        {\
     'data': 'data/train_860', \
     'tgt': 'data/train_860/pdfid.${subsampling}.tgt', \
     'batchsize': 32, 'chunk_width': 40, \
     'left_context': 5, 'right_context': 5 \
       },\
     ]" \
     `dirname ${chaindir}`/model${modelnum}
fi

# DECODING
if [ $stage -eq 15 ]; then
  # Echo Make graph if it does not exist
  if [ ! -f ${tree}/graph_tgsmall/HCLG.fst ]; then 
    ./utils/mkgraph.sh --self-loop-scale 1.0 \
      data/lang_test_tgsmall ${tree} ${tree}/graph_tgsmall
  fi

  # Prepare the test sets if not already done
  if [ ! -f data/dev_clean_fbank/feats.scp.dat ]; then
    ./local/prepare_test.sh --subsampling ${subsampling} --data ${unlabeled_data} 
  fi

  # Average models (This gives better performance)
  #average_models.py `dirname ${chaindir}`/model${modelnum} 80 60 160  
  for ds in $testsets; do 
    ./decode_nnet_pytorch.sh --min-lmwt 6 \
                           --max-lmwt 18 \
                           --skip-datadump true \
                           --modelname ${modelname} \
                           --acoustic-scale 1.0 \
                           --post-decode-acwt 10.0 \
                           --nj ${decode_nj} \
                           data/${ds}_fbank exp/model${modelnum} \
                           ${tree}/graph_tgsmall exp/model${modelnum}/decode_${modelname}_graph_${ds}
    echo ${decode_nj} > exp/model${modelnum}/decode_${modelname}_graph_${ds}/num_jobs
    ./steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_test_tg{small,large} \
      data/${ds}_fbank exp/model${modelnum}/decode_${modelname}_graph_${ds}{,_tglarge_rescored} 
  done
fi

# Generation
if [ $stage -eq 16 ]; then
  modeldir=`dirname ${chaindir}`/model${modelnum}
  gen_dir=${modeldir}/generate_cond_160.mdl
  mkdir -p ${gen_dir}
  generate_cmd="./utils/queue.pl --mem 2G --gpu 1 --config conf/gpu.conf ${gen_dir}/log"
  ${generate_cmd} generate_conditional_from_buffer.py \
    --gpu \
    --target 1084 1084 1084 1084 1084\
    --idim 80 --chunk-width 20 --left-context 10 --right-context 5 \
    --modeldir ${modeldir} --modelname 160.mdl \
    --dumpdir ${gen_dir} --batchsize 32
fi
