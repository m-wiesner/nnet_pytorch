#!/bin/bash

# DATA-level specifications. 
speech_data=/export/corpora5 #/PATH/TO/LIBRISPEECH/data
data=./corpus
data_url=www.openslr.org/resources/31
lm_url=www.openslr.org/resources/11

. ./cmd.sh
. ./path.sh

stage=0
subsampling=4
chaindir=exp/chain
model_dirname=model1
checkpoint=180_220.mdl
acwt=1.0
resume=
testsets="dev_clean dev_other test_clean test_other"
decode_nj=80
num_split=20 # number of splits for memory-mapped data for training
. ./utils/parse_options.sh

set -euo pipefail

tree=${chaindir}/tree
post_decode_acwt=`echo ${acwt} | awk '{print 10*$1}'`
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
  local/data_prep.sh ${speech_data}/LibriSpeech/train-clean-100 data/train_100h
  ./steps/make_mfcc.sh --cmd "$train_cmd" --nj 32 data/train_100h exp/make_mfcc/train_100h mfcc
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

  # Larger 3-gram LM rescoring
  #utils/build_const_arpa_lm.sh \
  #  data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge

  # 4-gram LM rescoring
  utils/build_const_arpa_lm.sh \
    data/local/lm/lm_fglarge.arpa.gz data/lang data/lang_test_fglarge

  steps/align_fmllr.sh --nj 5 --cmd "$train_cmd" \
    data/train_100h data/lang exp/tri3 exp/tri3_ali_train_100h
fi

if [ $stage -le 10 ]; then
  traindir=data/train_100h
  feat_affix=_fbank
  echo "Making features for nnet training ..."
  ./utils/copy_data_dir.sh ${traindir} ${traindir}${feat_affix}
  ./steps/make_fbank.sh --cmd "$train_cmd" --nj 32 ${traindir}${feat_affix}
  ./utils/fix_data_dir.sh ${traindir}${feat_affix}
  ./steps/compute_cmvn_stats.sh ${traindir}${feat_affix}
  ./utils/fix_data_dir.sh ${traindir}${feat_affix}

  echo "Dumping memory mapped features ..."
  ./local/split_memmap_data.sh data/train_100h_fbank ${num_split} 
fi


