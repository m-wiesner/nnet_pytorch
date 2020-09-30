#!/bin/bash

# This is based almost entirely on the Kaldi Librispeech recipe
# This recipe ASSUMES YOU HAVE DOWNLOADED the Librispeech data
# This is the training script. For decoding, see decode.sh

corpus_dir=/export/corpora5/LibriSpeech #/PATH/TO/LIBRISPEECH/data
data=./corpus
data_url=www.openslr.org/resources/31
lm_url=www.openslr.org/resources/11
mfccdir=mfcc

. ./cmd.sh
. ./path.sh

stage=0
subsampling=3
num_split=20

. ./utils/parse_options.sh

set -euo pipefail

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
fi

if [ $stage -le 2 ]; then
  # format the data as Kaldi data directories
  for part in train-clean-100; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $corpus_dir/$part data/$(echo $part | sed s/-/_/g)
  done
fi

if [ $stage -le 3 ]; then
  # spread the mfccs and fbanks over various machines, as this data-set is quite large.
  if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
    mfcc=$(basename mfccdir) # in case was absolute pathname (unlikely), get basename.
    utils/create_split_dir.pl /export/b{11,12,13}/$USER/kaldi-data/egs/librispeech/s5/$mfcc/storage \
     $mfccdir/storage
  fi
fi

if [ $stage -le 4 ]; then
  # Prepare the train_clean_100 subset 
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 32 data/train_clean_100 exp/make_mfcc/train_clean_100 mfcc
  utils/fix_data_dir.sh data/train_clean_100
  steps/compute_cmvn_stats.sh data/train_clean_100
  utils/fix_data_dir.sh data/train_clean_100

  utils/subset_data_dir.sh --shortest data/train_clean_100 500 data/train_500short
  utils/subset_data_dir.sh data/train_clean_100 5000 data/train_5k
  utils/subset_data_dir.sh data/train_clean_100 10000 data/train_10k
fi

# train a monophone system
if [ $stage -le 5 ]; then 
  steps/train_mono.sh --boost-silence 1.25 --nj 15 --cmd "$train_cmd" \
    data/train_500short data/lang_nosp exp/mono
  
  steps/align_si.sh --boost-silence 1.25 --nj 15 --cmd "$train_cmd" \
    data/train_5k data/lang_nosp exp/mono exp/mono_ali_train_5k
fi

# train a first delta + delta-delta triphone system on 5k utterances
if [ $stage -le 6 ]; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train_5k data/lang_nosp exp/mono_ali_train_5k exp/tri1

  steps/align_si.sh --nj 15 --cmd "$train_cmd" \
    data/train_10k data/lang_nosp exp/tri1 exp/tri1_ali_train_10k
fi

# train a first delta + delta-delta triphone system on 10k utterances
if [ $stage -le 7 ]; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2500 15000 data/train_10k data/lang_nosp exp/tri1_ali_train_10k exp/tri1b

  steps/align_si.sh --nj 20 --cmd "$train_cmd" \
    data/train_clean_100 data/lang_nosp exp/tri1b exp/tri1b_ali_train_clean_100
fi

# train an LDA+MLLT system.
if [ $stage -le 8 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 4200 40000 \
    data/train_clean_100 data/lang_nosp exp/tri1b_ali_train_clean_100 exp/tri2

  # Align utts using the tri2b model
  steps/align_si.sh --nj 20 --cmd "$train_cmd" --use-graphs true \
    data/train_clean_100 data/lang_nosp exp/tri2 exp/tri2_ali_train_clean_100
fi

# Train tri3, which is LDA+MLLT+SAT
if [ $stage -le 9 ]; then
  steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
    data/train_clean_100 data/lang_nosp exp/tri2_ali_train_clean_100 exp/tri3
fi

# Now we compute the pronunciation and silence probabilities from training data,
# and re-create the lang directory.
if [ $stage -le 10 ]; then
  steps/get_prons.sh --cmd "$train_cmd" \
    data/train_clean_100 data/lang_nosp exp/tri3
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp \
    exp/tri3/pron_counts_nowb.txt exp/tri3/sil_counts_nowb.txt \
    exp/tri3/pron_bigram_counts_nowb.txt data/local/dict

  utils/prepare_lang.sh data/local/dict \
    "<UNK>" data/local/lang_tmp data/lang

  local/format_lms.sh --src-dir data/lang data/local/lm

  utils/build_const_arpa_lm.sh \
    data/local/lm/lm_fglarge.arpa.gz data/lang data/lang_test_fglarge

  steps/align_fmllr.sh --nj 5 --cmd "$train_cmd" \
    data/train_clean_100 data/lang exp/tri3 exp/tri3_ali_train_clean_100
fi

if [ $stage -le 11 ]; then
  # add the "clean-360" subset to the mix ...
  local/data_prep.sh \
    $corpus_dir/train-clean-360 data/train_clean_360
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/train_clean_360 \
                     exp/make_mfcc/train_clean_360 $mfccdir
  steps/compute_cmvn_stats.sh \
    data/train_clean_360 exp/make_mfcc/train_clean_360 $mfccdir

  # ... and then combine the two sets into a 460 hour one
  utils/combine_data.sh \
    data/train_clean_460 data/train_clean_100 data/train_clean_360
fi

if [ $stage -le 12 ]; then
  # align the new, combined set, using the tri4b model
  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
                       data/train_clean_460 data/lang exp/tri3 exp/tri3_ali_clean_460

  # create a larger SAT model, trained on the 460 hours of data.
  steps/train_sat.sh  --cmd "$train_cmd" 5000 100000 \
                      data/train_clean_460 data/lang exp/tri3_ali_clean_460 exp/tri4b
fi

if [ $stage -le 13 ]; then
  # prepare the 500 hour subset.
  local/data_prep.sh \
    $corpus_dir/train-other-500 data/train_other_500
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/train_other_500 \
                     exp/make_mfcc/train_other_500 $mfccdir
  steps/compute_cmvn_stats.sh \
    data/train_other_500 exp/make_mfcc/train_other_500 $mfccdir

  # combine all the data
  utils/combine_data.sh \
    data/train_960 data/train_clean_460 data/train_other_500
fi

if [ $stage -le 14 ]; then
  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
                       data/train_960 data/lang exp/tri4b exp/tri4b_ali_960

  # train a SAT model on the 960 hour mixed data.  Use the train_quick.sh script
  # as it is faster.
  steps/train_quick.sh --cmd "$train_cmd" \
                       7000 150000 data/train_960 data/lang exp/tri4b_ali_960 exp/tri5b
fi

if [ $stage -le 15 ]; then
  traindir=data/train_960
  feat_affix=_fbank
  utils/copy_data_dir.sh ${traindir} ${traindir}${feat_affix}
  steps/make_fbank.sh --cmd "$train_cmd" --nj 40 ${traindir}${feat_affix} exp/make_fbank/train_960 fbank
  utils/fix_data_dir.sh ${traindir}${feat_affix}
  steps/compute_cmvn_stats.sh ${traindir}${feat_affix}
  utils/fix_data_dir.sh ${traindir}${feat_affix}

  echo "Dumping memory mapped features ..."
  split_memmap_data.sh ${traindir}${feat_affix} ${num_split} 
fi
